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
    geo_dim: int = 8           # geometry features [0:8]
    ssp_dim: int = 20          # SSP features [8:28]
    bathy_dim: int = 20        # bathymetry features [28:48]
    reserved_dim: int = 16     # reserved features [48:64]
    branch_dim: int = 256      # per-branch encoder width
    fusion_dim: int = 512      # fusion trunk width
    n_branch_layers: int = 3   # layers per branch encoder
    n_fusion_layers: int = 6   # layers in fusion trunk
    dropout: float = 0.02


class ResidualBlock(nn.Module):
    """Pre-norm residual block: LayerNorm -> Linear -> GELU -> Dropout -> Linear -> Add"""
    def __init__(self, dim, dropout=0.02):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * 2)
        self.fc2 = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm(x)
        h = F.gelu(self.fc1(h))
        h = self.dropout(h)
        h = self.fc2(h)
        return x + h


def make_branch_encoder(in_dim, hidden_dim, n_layers, dropout):
    """Build a small MLP encoder for a feature branch."""
    layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
    for _ in range(n_layers - 1):
        layers.extend([
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        ])
    return nn.Sequential(*layers)


class CrossBranchAttention(nn.Module):
    """Lightweight cross-attention: each branch attends to all branches."""
    def __init__(self, dim, n_heads=4, dropout=0.02):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, branches):
        """branches: (B, 3, dim) — 3 branch embeddings stacked as a sequence."""
        B, S, D = branches.shape
        h = self.norm(branches)
        qkv = self.qkv(h).reshape(B, S, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, S, D)
        return branches + self.proj(out)


class TLNet(nn.Module):
    """Branch-encoder + cross-attention + fusion trunk for TL prediction.

    1. Split input into 3 semantic groups (geometry, SSP, bathymetry)
    2. Encode each branch independently
    3. Cross-attention between branches (learn interactions)
    4. Concatenate + fusion trunk
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Branch encoders
        self.geo_encoder = make_branch_encoder(
            config.geo_dim, config.branch_dim, config.n_branch_layers, config.dropout)
        self.ssp_encoder = make_branch_encoder(
            config.ssp_dim, config.branch_dim, config.n_branch_layers, config.dropout)
        self.bathy_encoder = make_branch_encoder(
            config.bathy_dim, config.branch_dim, config.n_branch_layers, config.dropout)

        # Cross-branch attention (2 layers)
        self.cross_attn = nn.ModuleList([
            CrossBranchAttention(config.branch_dim, n_heads=4, dropout=config.dropout)
            for _ in range(2)
        ])

        # Fusion: project concatenated branch outputs to fusion dim
        total_branch = config.branch_dim * 3
        self.fusion_proj = nn.Linear(total_branch, config.fusion_dim)

        # Fusion trunk (residual blocks)
        self.fusion_blocks = nn.ModuleList([
            ResidualBlock(config.fusion_dim, config.dropout)
            for _ in range(config.n_fusion_layers)
        ])
        self.out_norm = nn.LayerNorm(config.fusion_dim)
        self.out_head = nn.Linear(config.fusion_dim, 1)

    def forward(self, x, targets=None):
        # Split input into branches
        geo = x[:, :8]
        ssp = x[:, 8:28]
        bathy = x[:, 28:48]

        # Encode each branch
        geo_h = self.geo_encoder(geo)    # (B, branch_dim)
        ssp_h = self.ssp_encoder(ssp)
        bathy_h = self.bathy_encoder(bathy)

        # Stack as sequence for cross-attention: (B, 3, branch_dim)
        branches = torch.stack([geo_h, ssp_h, bathy_h], dim=1)
        for attn_layer in self.cross_attn:
            branches = attn_layer(branches)

        # Flatten back and fuse
        h = branches.reshape(branches.size(0), -1)  # (B, 3*branch_dim)
        h = F.gelu(self.fusion_proj(h))

        # Fusion trunk
        for block in self.fusion_blocks:
            h = block(h)
        h = self.out_norm(h)
        pred = self.out_head(h).squeeze(-1)

        if targets is not None:
            return F.huber_loss(pred, targets, delta=5.0)
        return pred

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
BRANCH_DIM = 256          # per-branch encoder width
FUSION_DIM = 512          # fusion trunk width
N_BRANCH_LAYERS = 3       # layers per branch encoder
N_FUSION_LAYERS = 6       # layers in fusion trunk
DROPOUT = 0.02            # dropout rate

# Optimization
BATCH_SIZE = 2048         # training batch size
LEARNING_RATE = 3e-3      # peak learning rate
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
    branch_dim=BRANCH_DIM,
    fusion_dim=FUSION_DIM,
    n_branch_layers=N_BRANCH_LAYERS,
    n_fusion_layers=N_FUSION_LAYERS,
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
print(f"branch_dim:       {BRANCH_DIM}")
print(f"fusion_dim:       {FUSION_DIM}")
print(f"n_branch_layers:  {N_BRANCH_LAYERS}")
print(f"n_fusion_layers:  {N_FUSION_LAYERS}")
