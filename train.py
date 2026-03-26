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

from prepare import MAX_FEATURES, TIME_BUDGET as _TIME_BUDGET, FeatureStats, make_dataloader, evaluate_rmse

# Override time budget for Phase 2 with large data (4.4M samples can support longer training)
TIME_BUDGET = 1800  # 30 minutes (16M data supports long training without overfitting)

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


class CrossBranchBlock(nn.Module):
    """Transformer-style block: cross-attention + FFN, both with pre-norm residual."""
    def __init__(self, dim, n_heads=4, dropout=0.02):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        # Attention
        self.attn_norm = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_dropout = nn.Dropout(dropout)
        # FFN
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn1 = nn.Linear(dim, dim * 2)
        self.ffn2 = nn.Linear(dim * 2, dim)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, branches):
        """branches: (B, 3, dim) — 3 branch embeddings stacked as a sequence."""
        B, S, D = branches.shape
        # Self-attention with pre-norm residual
        h = self.attn_norm(branches)
        qkv = self.qkv(h).reshape(B, S, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        out = (attn @ v).transpose(1, 2).reshape(B, S, D)
        branches = branches + self.proj(out)
        # FFN with pre-norm residual
        h = self.ffn_norm(branches)
        h = F.gelu(self.ffn1(h))
        h = self.ffn_dropout(h)
        h = self.ffn2(h)
        branches = branches + h
        return branches


class TLNet(nn.Module):
    """Physics-informed branch-encoder + cross-attention + fusion trunk for TL.

    Architecture: Empirical formula (base) + Neural network (residual correction)
    1. Compute empirical TL from raw physics (spreading + Thorp absorption)
    2. Split input into 3 semantic groups (geometry, SSP, bathymetry)
    3. Encode each branch independently
    4. Cross-attention between branches (learn interactions)
    5. Fusion trunk predicts RESIDUAL = actual_TL - empirical_TL
    6. Final prediction = empirical_TL + residual
    """

    def __init__(self, config, feat_mean=None, feat_std=None):
        super().__init__()
        self.config = config

        # Store normalization stats as buffers (for denormalizing to compute empirical TL)
        if feat_mean is not None:
            self.register_buffer('feat_mean', feat_mean)
            self.register_buffer('feat_std', feat_std)

        # Branch encoders (geo gets +3 augmented features: freq*range, |depth_diff|, empirical_tl_norm)
        self.geo_encoder = make_branch_encoder(
            config.geo_dim + 3, config.branch_dim, config.n_branch_layers, config.dropout)
        self.ssp_encoder = make_branch_encoder(
            config.ssp_dim, config.branch_dim, config.n_branch_layers, config.dropout)
        self.bathy_encoder = make_branch_encoder(
            config.bathy_dim, config.branch_dim, config.n_branch_layers, config.dropout)

        # Cross-branch transformer blocks (3 layers)
        self.cross_attn = nn.ModuleList([
            CrossBranchBlock(config.branch_dim, n_heads=8, dropout=config.dropout)
            for _ in range(3)
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

    def compute_empirical_tl(self, x_norm):
        """Compute environment-aware empirical TL from normalized features.

        Full semi-empirical model with 6 physical effects:
        1. Geometric spreading (spherical → cylindrical transition)
        2. Thorp absorption (frequency-dependent)
        3. Lloyd mirror effect (shallow source/receiver interference)
        4. Bottom reflection loss (impedance-based)
        5. Sound channel ducting bonus (SOFAR channel detection)
        6. Surface duct effect (positive SSP gradient)

        All computed on GPU in batch, no loops.
        """
        # Denormalize to get raw physical values
        raw = x_norm * self.feat_std + self.feat_mean

        freq_hz = raw[:, 0].clamp(min=10.0)
        src_depth = raw[:, 1].clamp(min=0.1)
        rcv_depth = raw[:, 2].clamp(min=0.1)
        range_km = raw[:, 3].clamp(min=0.001)
        water_depth_src = raw[:, 4].clamp(min=1.0)
        water_depth_rcv = raw[:, 5].clamp(min=1.0)
        bottom_ss = raw[:, 6].clamp(min=1400.0)
        bottom_density = raw[:, 7].clamp(min=1.0)
        ssp = raw[:, 8:28]           # (B, 20) sound speed profile values
        # bathy = raw[:, 28:48]      # not needed for empirical TL

        range_m = range_km * 1000.0
        avg_depth = (water_depth_src + water_depth_rcv) * 0.5

        # === 1. Geometric spreading ===
        transition_r = avg_depth.clamp(min=10.0)
        log10_rm = torch.log10(range_m)
        log10_tr = torch.log10(transition_r)
        spreading = torch.where(
            range_m <= transition_r,
            20.0 * log10_rm,
            20.0 * log10_tr + 10.0 * (log10_rm - log10_tr)
        )

        # === 2. Absorption (Thorp) ===
        f_khz = freq_hz / 1000.0
        f2 = f_khz ** 2
        alpha = 0.11 * f2 / (1.0 + f2) + 44.0 * f2 / (4100.0 + f2) + 2.75e-4 * f2 + 0.003
        absorption = alpha * range_km

        # === 3. Lloyd mirror effect ===
        wavelength = 1500.0 / freq_hz
        path_diff = 2.0 * src_depth * rcv_depth / range_m
        lloyd_raw = 6.0 * (1.0 - path_diff / wavelength)
        # Only active for shallow sources, far range, path_diff < wavelength
        lloyd_mask = (src_depth < 100.0) & (range_m > 100.0) & (path_diff < wavelength)
        lloyd_mirror = torch.where(lloyd_mask, lloyd_raw.clamp(min=0.0, max=6.0), torch.zeros_like(lloyd_raw))

        # === 4. Bottom reflection loss ===
        grazing_angle = torch.atan2(torch.max(src_depth, rcv_depth), range_m)
        avg_water_ss = ssp[:, :5].mean(dim=1)  # shallow sound speed

        # Critical angle
        ss_ratio = (avg_water_ss / bottom_ss).clamp(min=0.0, max=1.0)
        critical_angle = torch.asin(ss_ratio)

        # Number of bottom bounces
        tan_ga = torch.tan(grazing_angle.clamp(min=0.01))
        bounce_spacing = (2.0 * avg_depth / tan_ga).clamp(min=1.0)
        n_bounces = (range_m / bounce_spacing).clamp(min=0.0)
        n_bounces = torch.min(n_bounces, range_km * 2.0)

        # Loss per bounce: small below critical angle, larger above
        impedance_ratio = (bottom_density * bottom_ss) / (1.0 * avg_water_ss)
        loss_above = (2.0 + 3.0 * (1.0 - 1.0 / impedance_ratio)).clamp(min=0.5, max=8.0)
        loss_below = torch.full_like(loss_above, 0.5)
        loss_per_bounce = torch.where(grazing_angle < critical_angle, loss_below, loss_above)

        bottom_loss = (n_bounces * loss_per_bounce).clamp(max=30.0)
        # Only significant for non-trivial grazing angles
        bottom_loss = torch.where(grazing_angle > 0.01, bottom_loss, torch.zeros_like(bottom_loss))

        # === 5. Sound channel ducting bonus ===
        # Detect SOFAR channel: SSP minimum not at surface/bottom
        ssp_min_val, ssp_min_idx = ssp.min(dim=1)
        ssp_max_val = ssp.max(dim=1).values
        ssp_range = ssp_max_val - ssp_min_val

        # SSP_DEPTHS approximate indices: channel exists if min is interior (idx 1-18)
        channel_exists = (ssp_range > 10.0) & (ssp_min_idx > 0) & (ssp_min_idx < 19)

        # Approximate channel axis depth from index (rough: idx * 250m average)
        # SSP_DEPTHS = [0,25,50,75,100,150,200,300,500,700,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500]
        ssp_depth_lut = torch.tensor(
            [0,25,50,75,100,150,200,300,500,700,1000,1500,2000,2500,3000,3500,4000,4500,5000,5500],
            dtype=torch.float32, device=x_norm.device)
        channel_depth = ssp_depth_lut[ssp_min_idx.clamp(0, 19)]

        channel_width = avg_depth * 0.5
        src_in_ch = (src_depth - channel_depth).abs() < channel_width
        rcv_in_ch = (rcv_depth - channel_depth).abs() < channel_width
        both_in_channel = channel_exists & src_in_ch & rcv_in_ch

        channel_strength = (ssp_range / 30.0).clamp(max=1.0)
        duct_raw = channel_strength * 5.0 * torch.log10((range_m / transition_r).clamp(min=1.0))
        duct_bonus = torch.where(both_in_channel, duct_raw.clamp(min=0.0, max=15.0), torch.zeros_like(duct_raw))

        # === 6. Surface duct effect ===
        # Positive SSP gradient near surface + shallow source & receiver
        surf_gradient = (ssp[:, 2] - ssp[:, 0]) / 50.0  # over ~50m depth span
        surf_duct_active = (surf_gradient > 0.01) & (src_depth < 50.0) & (rcv_depth < 50.0)
        surf_duct_raw = (surf_gradient * 100.0).clamp(max=5.0) * torch.log10(range_km.clamp(min=1.0))
        surface_duct = torch.where(surf_duct_active, surf_duct_raw.clamp(min=0.0, max=10.0), torch.zeros_like(surf_duct_raw))

        # === Combine all effects ===
        empirical_tl = spreading + absorption + lloyd_mirror + bottom_loss - duct_bonus - surface_duct
        empirical_tl = empirical_tl.clamp(min=10.0, max=200.0)

        return empirical_tl

    def forward(self, x, targets=None):
        # Compute empirical TL (physics baseline)
        empirical_tl = self.compute_empirical_tl(x)

        # Split input into branches
        geo = x[:, :8]
        ssp = x[:, 8:28]
        bathy = x[:, 28:48]

        # Augment geo with physics-derived features:
        freq_feat = x[:, 0:1]
        range_feat = x[:, 3:4]
        depth_diff = (x[:, 1:2] - x[:, 2:3]).abs()
        fr_interact = freq_feat * range_feat
        # Also feed normalized empirical TL as a feature (helps the NN know the baseline)
        empirical_norm = (empirical_tl / 100.0).unsqueeze(1)  # rough normalization ~0-2 range
        geo_aug = torch.cat([geo, fr_interact, depth_diff, empirical_norm], dim=1)  # 8+3=11

        # Encode each branch
        geo_h = self.geo_encoder(geo_aug)    # (B, branch_dim)
        ssp_h = self.ssp_encoder(ssp)
        bathy_h = self.bathy_encoder(bathy)

        # Stack as sequence for cross-attention: (B, 3, branch_dim)
        branches = torch.stack([geo_h, ssp_h, bathy_h], dim=1)
        for attn_layer in self.cross_attn:
            branches = attn_layer(branches)

        # Flatten back and fuse
        h = branches.reshape(branches.size(0), -1)  # (B, 3*branch_dim)
        h = F.gelu(self.fusion_proj(h))

        # Fusion trunk — predicts RESIDUAL correction
        for block in self.fusion_blocks:
            h = block(h)
        h = self.out_norm(h)
        residual = self.out_head(h).squeeze(-1)

        # Final prediction = empirical baseline + learned correction
        pred = empirical_tl + residual

        if targets is not None:
            return F.huber_loss(pred, targets, delta=5.0)
        return pred

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

# Model architecture
BRANCH_DIM = 512          # per-branch encoder width
FUSION_DIM = 1024         # fusion trunk width
N_BRANCH_LAYERS = 3       # layers per branch encoder
N_FUSION_LAYERS = 6       # layers in fusion trunk
DROPOUT = 0.02            # dropout rate

# Optimization
BATCH_SIZE = 4096         # training batch size
LEARNING_RATE = 3e-3      # peak learning rate
WEIGHT_DECAY = 5e-5       # AdamW weight decay
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

# Pass normalization stats to model for empirical TL computation
feat_mean = stats.mean.to(device)
feat_std = stats.std.to(device)
model = TLNet(config, feat_mean=feat_mean, feat_std=feat_std).to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {num_params:,} ({num_params/1000:.1f}K)")
print(f"Architecture: Empirical formula (spreading+Thorp) + NN residual correction")

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

    # Gradient clipping + optimizer step
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
