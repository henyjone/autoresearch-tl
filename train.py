"""
Autoresearch-TL training script. Single-GPU, single-file.
Neural network surrogate model for underwater acoustic transmission loss prediction.
Usage: uv run train.py
"""

import os
import gc
import math
import time
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import MAX_FEATURES, TIME_BUDGET, FeatureStats, make_dataloader, evaluate_rmse

# ---------------------------------------------------------------------------
# TL Prediction Model
# ---------------------------------------------------------------------------

@dataclass
class TLNetConfig:
    n_features: int = 64       # MAX_FEATURES from prepare.py
    hidden_dim: int = 256
    n_layers: int = 4
    dropout: float = 0.1


class TLNet(nn.Module):
    """Baseline MLP for transmission loss prediction.

    Input:  normalized feature vector (B, MAX_FEATURES)
    Output: predicted TL in dB (B,)

    Architecture: Linear -> GELU -> Dropout -> ... -> Linear(1)
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        layers = []
        in_dim = config.n_features
        for i in range(config.n_layers):
            layers.append(nn.Linear(in_dim, config.hidden_dim))
            layers.append(nn.GELU())
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))
            in_dim = config.hidden_dim
        layers.append(nn.Linear(config.hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, targets=None):
        """
        Args:
            x: (B, MAX_FEATURES) normalized input features
            targets: (B,) TL values in dB, or None

        Returns:
            If targets is not None: scalar MSE loss
            If targets is None: predictions (B,)
        """
        pred = self.net(x).squeeze(-1)  # (B,)
        if targets is not None:
            return F.mse_loss(pred, targets)
        return pred

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
HIDDEN_DIM = 256          # hidden layer width
N_LAYERS = 4              # number of hidden layers
DROPOUT = 0.1             # dropout rate

# Optimization
BATCH_SIZE = 1024         # training batch size
LEARNING_RATE = 1e-3      # peak learning rate
WEIGHT_DECAY = 1e-4       # AdamW weight decay
ADAM_BETAS = (0.9, 0.999) # Adam beta parameters
WARMUP_RATIO = 0.05       # fraction of time for LR warmup
WARMDOWN_RATIO = 0.3      # fraction of time for LR cooldown
FINAL_LR_FRAC = 0.01      # final LR as fraction of peak

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(42)
torch.cuda.manual_seed(42)
device = torch.device("cuda")

# Load normalization stats
stats = FeatureStats.from_data()
print(f"Feature stats loaded (target mean={stats.target_mean:.1f} dB, std={stats.target_std:.1f} dB)")

# Build model
config = TLNetConfig(
    n_features=MAX_FEATURES,
    hidden_dim=HIDDEN_DIM,
    n_layers=N_LAYERS,
    dropout=DROPOUT,
)
print(f"Model config: {asdict(config)}")

model = TLNet(config).to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {num_params:,} ({num_params/1000:.1f}K)")

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    betas=ADAM_BETAS,
    weight_decay=WEIGHT_DECAY,
)

# Compile model for faster training (requires Triton, skip on Windows)
import sys
if sys.platform != "win32":
    model = torch.compile(model, dynamic=False)

# Dataloader
train_loader = make_dataloader(stats, BATCH_SIZE, "train")
x, y, epoch = next(train_loader)  # prefetch first batch

print(f"Time budget: {TIME_BUDGET}s")
print(f"Batch size: {BATCH_SIZE}")

# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def get_lr_multiplier(progress):
    """LR schedule: linear warmup -> constant -> cosine warmdown."""
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    elif progress < 1.0 - WARMDOWN_RATIO:
        return 1.0
    else:
        warmdown_progress = (1.0 - progress) / WARMDOWN_RATIO
        # Cosine decay
        return FINAL_LR_FRAC + (1.0 - FINAL_LR_FRAC) * 0.5 * (1 + math.cos(math.pi * (1 - warmdown_progress)))

# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0

# Progress log file (clean, appendable, one line per LOG_INTERVAL steps)
LOG_INTERVAL = 500
progress_log_path = os.path.join(os.path.dirname(__file__), "progress.log")
with open(progress_log_path, "w") as f:
    f.write("step\tprogress\tloss\trmse_approx\tlr\tepoch\tremaining_s\n")

while True:
    torch.cuda.synchronize()
    t0 = time.time()

    # Forward + backward
    model.train()
    loss = model(x, y)
    train_loss = loss.detach()
    loss.backward()

    # Optimizer step
    optimizer.step()
    model.zero_grad(set_to_none=True)

    # Prefetch next batch
    x, y, epoch = next(train_loader)

    # Progress and LR schedule
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lrm = get_lr_multiplier(progress)
    for group in optimizer.param_groups:
        group["lr"] = LEARNING_RATE * lrm

    train_loss_f = train_loss.item()

    # Fast fail: abort if loss is exploding or NaN
    if math.isnan(train_loss_f) or train_loss_f > 1e6:
        print("FAIL")
        exit(1)

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step > 5:
        total_training_time += dt

    # Logging
    ema_beta = 0.95
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss_f
    debiased_smooth_loss = smooth_train_loss / (1 - ema_beta**(step + 1))
    pct_done = 100 * progress
    remaining = max(0, TIME_BUDGET - total_training_time)
    rmse_approx = math.sqrt(max(debiased_smooth_loss, 0))

    print(f"\rstep {step:05d} ({pct_done:.1f}%) | loss: {debiased_smooth_loss:.4f} | ~rmse: {rmse_approx:.2f} dB | lr: {LEARNING_RATE * lrm:.6f} | dt: {dt*1000:.0f}ms | epoch: {epoch} | remaining: {remaining:.0f}s    ", end="", flush=True)

    # Write to progress log periodically
    if step % LOG_INTERVAL == 0:
        with open(progress_log_path, "a") as f:
            f.write(f"{step}\t{pct_done:.1f}\t{debiased_smooth_loss:.4f}\t{rmse_approx:.2f}\t{LEARNING_RATE * lrm:.6f}\t{epoch}\t{remaining:.0f}\n")

    # GC management
    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()

    step += 1

    # Time's up — but only stop after warmup steps
    if step > 5 and total_training_time >= TIME_BUDGET:
        break

print()  # newline after \r training log

total_samples = step * BATCH_SIZE

# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------

model.eval()
val_rmse = evaluate_rmse(model, stats, BATCH_SIZE)

# Final summary
t_end = time.time()
startup_time = t_start_training - t_start
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"val_rmse:         {val_rmse:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"total_samples_M:  {total_samples / 1e6:.1f}")
print(f"num_steps:        {step}")
print(f"num_params_K:     {num_params / 1000:.1f}")
print(f"hidden_dim:       {HIDDEN_DIM}")
print(f"n_layers:         {N_LAYERS}")
