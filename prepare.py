"""
Autoresearch-TL data preparation and evaluation.
Generates underwater acoustic transmission loss (TL) training data,
provides dataloading utilities and the fixed evaluation metric.

Usage:
    python prepare.py                  # generate Phase 1 synthetic data
    python prepare.py --num-samples 50000  # smaller dataset for testing

Data is stored in ~/.cache/autoresearch-tl/.
This file is FIXED — do not modify during agent experimentation.
"""

import os
import sys
import math
import time
import argparse
from dataclasses import dataclass

import numpy as np
import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MAX_FEATURES = 64            # fixed input feature vector length
TIME_BUDGET = 300            # training time budget in seconds (5 minutes)
EVAL_SAMPLES = 50_000        # number of validation samples for eval

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-tl")
DATA_DIR = os.path.join(CACHE_DIR, "data")

# Standard depths for SSP sampling (meters) — 20 levels
SSP_DEPTHS = np.array([
    0, 25, 50, 75, 100, 150, 200, 300, 500, 700,
    1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500
], dtype=np.float64)

# Feature vector layout
# [0]     frequency_hz (log10 scaled in normalization)
# [1]     source_depth_m
# [2]     receiver_depth_m
# [3]     range_km
# [4]     water_depth_source_m
# [5]     water_depth_receiver_m
# [6]     bottom_soundspeed_mps
# [7]     bottom_density_gcc
# [8:28]  ssp_values (20 depths)
# [28:48] bathymetry_profile (20 range points)
# [48:64] reserved (zeros)

IDX_FREQ = 0
IDX_SRC_DEPTH = 1
IDX_RCV_DEPTH = 2
IDX_RANGE = 3
IDX_WATER_DEPTH_SRC = 4
IDX_WATER_DEPTH_RCV = 5
IDX_BOTTOM_SS = 6
IDX_BOTTOM_DENSITY = 7
IDX_SSP_START = 8
IDX_SSP_END = 28
IDX_BATHY_START = 28
IDX_BATHY_END = 48
IDX_RESERVED_START = 48
IDX_RESERVED_END = 64

# Bottom type presets (Hamilton-Bachman empirical values)
BOTTOM_TYPES = {
    "sand":    {"soundspeed": 1650.0, "density": 1.9},
    "silt":    {"soundspeed": 1550.0, "density": 1.7},
    "clay":    {"soundspeed": 1500.0, "density": 1.5},
    "gravel":  {"soundspeed": 1800.0, "density": 2.0},
    "rock":    {"soundspeed": 2500.0, "density": 2.5},
}

# ---------------------------------------------------------------------------
# Physics utilities
# ---------------------------------------------------------------------------

def thorp_absorption(freq_hz):
    """Thorp absorption coefficient in dB/km.

    Valid for frequencies 100 Hz - 10 kHz.
    Below 100 Hz, uses low-frequency approximation.
    """
    f_khz = freq_hz / 1000.0
    f2 = f_khz ** 2
    # Thorp formula (dB/km)
    alpha = (0.11 * f2 / (1 + f2)) + (44 * f2 / (4100 + f2)) + 2.75e-4 * f2 + 0.003
    return alpha


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two points in km."""
    R = 6371.0  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2)**2
    return R * 2 * math.asin(math.sqrt(a))


def generate_ssp_profile(profile_type, water_depth, rng):
    """Generate a sound speed profile at SSP_DEPTHS.

    Returns array of 20 sound speed values (m/s).
    Values below the seafloor are filled with the deepest valid value.

    Profile types:
        isovelocity:  constant ~1500 m/s with small noise
        thermocline:  warm surface layer + sharp gradient + deep isothermal
        deep_channel: minimum at ~1000m (SOFAR channel)
        arctic:       minimum at surface, increasing with depth
        shallow:      linear gradient for shallow water
    """
    ssp = np.zeros(20, dtype=np.float64)

    if profile_type == "isovelocity":
        base = 1490 + rng.uniform(-10, 10)
        ssp[:] = base + rng.normal(0, 0.5, 20)

    elif profile_type == "thermocline":
        # Surface mixed layer: warm, high sound speed
        surface_ss = 1520 + rng.uniform(-10, 10)
        deep_ss = 1490 + rng.uniform(-5, 5)
        thermocline_depth = rng.uniform(50, 200)  # meters
        thermocline_width = rng.uniform(50, 150)

        for i, d in enumerate(SSP_DEPTHS):
            if d < thermocline_depth:
                ssp[i] = surface_ss
            elif d < thermocline_depth + thermocline_width:
                frac = (d - thermocline_depth) / thermocline_width
                ssp[i] = surface_ss + (deep_ss - surface_ss) * frac
            else:
                # Below thermocline: slight increase with depth (pressure effect)
                ssp[i] = deep_ss + 0.017 * (d - thermocline_depth - thermocline_width)
        ssp += rng.normal(0, 0.3, 20)

    elif profile_type == "deep_channel":
        # SOFAR channel: minimum around 700-1300m
        channel_depth = rng.uniform(700, 1300)
        surface_ss = 1530 + rng.uniform(-10, 5)
        min_ss = 1485 + rng.uniform(-5, 5)

        for i, d in enumerate(SSP_DEPTHS):
            if d <= channel_depth:
                # Decrease to minimum (thermocline effect dominates)
                frac = d / channel_depth
                ssp[i] = surface_ss + (min_ss - surface_ss) * frac
            else:
                # Increase below minimum (pressure effect dominates)
                ssp[i] = min_ss + 0.017 * (d - channel_depth)
        ssp += rng.normal(0, 0.3, 20)

    elif profile_type == "arctic":
        # Cold surface, sound speed increases with depth
        surface_ss = 1440 + rng.uniform(-5, 5)
        for i, d in enumerate(SSP_DEPTHS):
            ssp[i] = surface_ss + 0.016 * d
        ssp += rng.normal(0, 0.3, 20)

    elif profile_type == "shallow":
        # Simple linear gradient for shallow water
        surface_ss = 1510 + rng.uniform(-15, 15)
        gradient = rng.uniform(-0.05, 0.05)  # m/s per meter depth
        for i, d in enumerate(SSP_DEPTHS):
            ssp[i] = surface_ss + gradient * d
        ssp += rng.normal(0, 0.2, 20)

    # Clip values below seafloor to deepest valid value
    deepest_valid = 0
    for i, d in enumerate(SSP_DEPTHS):
        if d <= water_depth:
            deepest_valid = i
    if deepest_valid < 19:
        ssp[deepest_valid+1:] = ssp[deepest_valid]

    # Ensure physically reasonable range
    ssp = np.clip(ssp, 1400, 1600)

    return ssp


def generate_bathymetry_profile(profile_type, water_depth_src, water_depth_rcv, n_points, rng):
    """Generate bathymetry along propagation path (n_points evenly spaced).

    Returns array of water depths in meters.

    Profile types:
        flat:        constant depth
        slope:       linear change from source to receiver depth
        shelf_break: continental shelf with steep drop-off
        seamount:    mid-path elevation
        rough:       irregular bottom with random perturbations
    """
    bathy = np.zeros(n_points, dtype=np.float64)
    t = np.linspace(0, 1, n_points)

    if profile_type == "flat":
        avg_depth = (water_depth_src + water_depth_rcv) / 2
        bathy[:] = avg_depth + rng.normal(0, avg_depth * 0.02, n_points)

    elif profile_type == "slope":
        bathy = water_depth_src + (water_depth_rcv - water_depth_src) * t
        bathy += rng.normal(0, abs(water_depth_rcv - water_depth_src) * 0.05 + 5, n_points)

    elif profile_type == "shelf_break":
        shelf_depth = rng.uniform(100, 300)
        deep_depth = rng.uniform(2000, 4000)
        break_pos = rng.uniform(0.2, 0.5)
        break_width = rng.uniform(0.05, 0.15)
        for i, ti in enumerate(t):
            if ti < break_pos:
                bathy[i] = shelf_depth
            elif ti < break_pos + break_width:
                frac = (ti - break_pos) / break_width
                bathy[i] = shelf_depth + (deep_depth - shelf_depth) * frac
            else:
                bathy[i] = deep_depth
        bathy += rng.normal(0, 20, n_points)

    elif profile_type == "seamount":
        base_depth = (water_depth_src + water_depth_rcv) / 2
        mount_pos = rng.uniform(0.3, 0.7)
        mount_width = rng.uniform(0.1, 0.25)
        mount_height = rng.uniform(base_depth * 0.3, base_depth * 0.7)
        for i, ti in enumerate(t):
            dist = abs(ti - mount_pos) / mount_width
            if dist < 1:
                bathy[i] = base_depth - mount_height * (1 - dist**2)
            else:
                bathy[i] = base_depth
        bathy += rng.normal(0, 15, n_points)

    elif profile_type == "rough":
        base = water_depth_src + (water_depth_rcv - water_depth_src) * t
        roughness = rng.uniform(0.05, 0.15) * np.mean([water_depth_src, water_depth_rcv])
        noise = np.cumsum(rng.normal(0, roughness / np.sqrt(n_points), n_points))
        noise -= np.linspace(noise[0], noise[-1], n_points)  # detrend
        bathy = base + noise

    bathy = np.clip(bathy, 10, 6000)
    return bathy


def compute_tl_analytical(freq_hz, src_depth, rcv_depth, range_km,
                          water_depth_src, water_depth_rcv,
                          bottom_ss, bottom_density,
                          ssp_values, bathymetry, rng):
    """Compute analytical transmission loss (dB) using simplified physics.

    Combines:
    1. Geometric spreading (spherical + transition to cylindrical)
    2. Thorp absorption
    3. Lloyd mirror effect (shallow source/receiver interference)
    4. Bottom reflection loss
    5. Sound channel ducting bonus (when SSP has minimum)
    6. Surface duct effect

    Returns TL in dB (positive value, higher = more loss).
    """
    range_m = range_km * 1000.0
    if range_m < 1.0:
        range_m = 1.0

    # 1. Geometric spreading
    # Transition from spherical to cylindrical at ~water_depth distance
    avg_depth = (water_depth_src + water_depth_rcv) / 2
    transition_range = avg_depth  # meters
    if range_m <= transition_range:
        spreading = 20.0 * np.log10(range_m)
    else:
        spreading = 20.0 * np.log10(transition_range) + 10.0 * np.log10(range_m / transition_range)

    # 2. Absorption (Thorp)
    alpha = thorp_absorption(freq_hz)
    absorption = alpha * range_km

    # 3. Lloyd mirror effect
    # Interference between direct and surface-reflected paths
    lloyd_mirror = 0.0
    wavelength = 1500.0 / freq_hz
    if src_depth < 100 and range_m > 100:
        # Effective source depth causes destructive interference
        path_diff = 2 * src_depth * rcv_depth / range_m
        if path_diff < wavelength:
            # In the shadow zone of the Lloyd mirror
            lloyd_mirror = max(0, 6.0 * (1.0 - path_diff / wavelength))

    # 4. Bottom reflection loss
    # Simplified: more loss for softer bottoms and steeper angles
    bottom_loss = 0.0
    grazing_angle = np.arctan2(max(src_depth, rcv_depth), range_m)
    if grazing_angle > 0.01:  # non-negligible bottom interaction
        # Critical angle based on bottom/water sound speed ratio
        avg_water_ss = np.mean(ssp_values[:5])  # use shallow SSP
        if bottom_ss > avg_water_ss:
            critical_angle = np.arcsin(avg_water_ss / bottom_ss)
        else:
            critical_angle = np.pi / 2

        # Number of bottom bounces (rough estimate)
        if avg_depth > 0:
            n_bounces = range_m / (2 * avg_depth / np.tan(max(grazing_angle, 0.01)))
            n_bounces = max(0, min(n_bounces, range_km * 2))
        else:
            n_bounces = 0

        # Loss per bounce depends on angle relative to critical angle
        if grazing_angle < critical_angle:
            loss_per_bounce = 0.5 + rng.uniform(0, 0.5)  # small loss below critical angle
        else:
            # Rayleigh reflection coefficient approximation
            impedance_ratio = (bottom_density * bottom_ss) / (1.0 * avg_water_ss)
            loss_per_bounce = 2.0 + 3.0 * (1.0 - 1.0/impedance_ratio)
            loss_per_bounce = max(0.5, min(loss_per_bounce, 8.0))

        bottom_loss = n_bounces * loss_per_bounce
        bottom_loss = min(bottom_loss, 30.0)  # cap bottom loss contribution

    # 5. Sound channel ducting bonus
    # If SSP has a minimum and source/receiver are near it, TL is reduced
    duct_bonus = 0.0
    ssp_min_idx = np.argmin(ssp_values)
    ssp_min_depth = SSP_DEPTHS[ssp_min_idx]
    ssp_range = np.max(ssp_values) - np.min(ssp_values)

    if ssp_range > 10 and ssp_min_idx > 0 and ssp_min_idx < 19:
        # Channel exists — check if source and receiver are in the channel
        channel_width = avg_depth * 0.5  # rough channel half-width
        src_in_channel = abs(src_depth - ssp_min_depth) < channel_width
        rcv_in_channel = abs(rcv_depth - ssp_min_depth) < channel_width

        if src_in_channel and rcv_in_channel:
            # Reduce spreading from spherical toward cylindrical
            channel_strength = min(1.0, ssp_range / 30.0)
            duct_bonus = channel_strength * 5.0 * np.log10(max(range_m / transition_range, 1.0))
            duct_bonus = max(0, min(duct_bonus, 15.0))

    # 6. Surface duct effect (positive SSP gradient near surface)
    surface_duct = 0.0
    if ssp_values[0] < ssp_values[2] and src_depth < 50 and rcv_depth < 50:
        gradient = (ssp_values[2] - ssp_values[0]) / SSP_DEPTHS[2]
        if gradient > 0.01:
            surface_duct = min(5.0, gradient * 100) * np.log10(max(range_km, 1.0))
            surface_duct = max(0, min(surface_duct, 10.0))

    # Combine
    tl = spreading + absorption + lloyd_mirror + bottom_loss - duct_bonus - surface_duct

    # Add small random perturbation to simulate unmodeled effects
    tl += rng.normal(0, 1.5)

    # Physical bounds
    tl = max(10.0, min(tl, 200.0))

    return tl


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_single_sample(rng):
    """Generate one (feature_vector, tl_value) sample.

    Returns:
        features: np.ndarray of shape (MAX_FEATURES,)
        tl: float, transmission loss in dB
    """
    features = np.zeros(MAX_FEATURES, dtype=np.float64)

    # Random scenario parameters
    freq = 10 ** rng.uniform(1.0, 4.0)  # 10 Hz to 10 kHz (log-uniform)

    # Choose water depth regime
    regime = rng.choice(["shallow", "medium", "deep"], p=[0.3, 0.4, 0.3])
    if regime == "shallow":
        water_depth_src = rng.uniform(20, 200)
    elif regime == "medium":
        water_depth_src = rng.uniform(200, 2000)
    else:
        water_depth_src = rng.uniform(2000, 5500)

    # Source and receiver depths (must be within water column)
    src_depth = rng.uniform(5, min(300, water_depth_src - 5))
    rcv_depth = rng.uniform(5, min(water_depth_src, water_depth_src - 5))

    # Range
    range_km = 10 ** rng.uniform(0.0, 2.3)  # 1 km to ~200 km (log-uniform)

    # Bottom type
    bottom_type = rng.choice(list(BOTTOM_TYPES.keys()))
    bottom_ss = BOTTOM_TYPES[bottom_type]["soundspeed"] + rng.normal(0, 20)
    bottom_density = BOTTOM_TYPES[bottom_type]["density"] + rng.normal(0, 0.1)

    # SSP profile
    ssp_type = rng.choice(["isovelocity", "thermocline", "deep_channel", "arctic", "shallow"],
                          p=[0.15, 0.30, 0.25, 0.15, 0.15])
    ssp_values = generate_ssp_profile(ssp_type, water_depth_src, rng)

    # Bathymetry profile
    # Receiver water depth may differ from source (range-dependent bathymetry)
    bathy_type = rng.choice(["flat", "slope", "shelf_break", "seamount", "rough"],
                            p=[0.25, 0.25, 0.20, 0.15, 0.15])
    if bathy_type == "flat":
        water_depth_rcv = water_depth_src + rng.normal(0, water_depth_src * 0.05)
    elif bathy_type == "slope":
        water_depth_rcv = water_depth_src * rng.uniform(0.3, 3.0)
    elif bathy_type == "shelf_break":
        water_depth_rcv = rng.uniform(1500, 4500) if water_depth_src < 500 else rng.uniform(50, 300)
    elif bathy_type == "seamount":
        water_depth_rcv = water_depth_src + rng.normal(0, water_depth_src * 0.1)
    else:
        water_depth_rcv = water_depth_src + rng.normal(0, water_depth_src * 0.15)
    water_depth_rcv = max(20, min(water_depth_rcv, 6000))

    # Ensure receiver depth is valid
    rcv_depth = min(rcv_depth, water_depth_rcv - 5)
    rcv_depth = max(5, rcv_depth)

    bathy_profile = generate_bathymetry_profile(
        bathy_type, water_depth_src, water_depth_rcv, 20, rng)

    # Compute TL
    tl = compute_tl_analytical(
        freq, src_depth, rcv_depth, range_km,
        water_depth_src, water_depth_rcv,
        bottom_ss, bottom_density,
        ssp_values, bathy_profile, rng)

    # Pack feature vector
    features[IDX_FREQ] = freq
    features[IDX_SRC_DEPTH] = src_depth
    features[IDX_RCV_DEPTH] = rcv_depth
    features[IDX_RANGE] = range_km
    features[IDX_WATER_DEPTH_SRC] = water_depth_src
    features[IDX_WATER_DEPTH_RCV] = water_depth_rcv
    features[IDX_BOTTOM_SS] = bottom_ss
    features[IDX_BOTTOM_DENSITY] = bottom_density
    features[IDX_SSP_START:IDX_SSP_END] = ssp_values
    features[IDX_BATHY_START:IDX_BATHY_END] = bathy_profile
    # features[IDX_RESERVED_START:IDX_RESERVED_END] = 0  (already zeros)

    return features, tl


def generate_dataset(n_samples, seed=42):
    """Generate a complete dataset of (features, tl) pairs.

    Returns:
        features: np.ndarray of shape (n_samples, MAX_FEATURES)
        targets: np.ndarray of shape (n_samples,)
    """
    rng = np.random.default_rng(seed)
    features = np.zeros((n_samples, MAX_FEATURES), dtype=np.float64)
    targets = np.zeros(n_samples, dtype=np.float64)

    for i in range(n_samples):
        features[i], targets[i] = generate_single_sample(rng)
        if (i + 1) % 50000 == 0:
            print(f"  Generated {i+1:,}/{n_samples:,} samples")

    return features, targets


def generate_phase1_data(n_train=200_000, n_val=50_000):
    """Generate Phase 1 synthetic TL data and save to disk."""
    os.makedirs(DATA_DIR, exist_ok=True)

    train_path = os.path.join(DATA_DIR, "train.npz")
    val_path = os.path.join(DATA_DIR, "val.npz")
    stats_path = os.path.join(DATA_DIR, "stats.npz")

    if os.path.exists(train_path) and os.path.exists(val_path) and os.path.exists(stats_path):
        print(f"Data: already generated at {DATA_DIR}")
        return

    print(f"Generating {n_train:,} training samples...")
    t0 = time.time()
    train_features, train_targets = generate_dataset(n_train, seed=42)
    t1 = time.time()
    print(f"  Training data generated in {t1-t0:.1f}s")

    print(f"Generating {n_val:,} validation samples...")
    val_features, val_targets = generate_dataset(n_val, seed=12345)
    t2 = time.time()
    print(f"  Validation data generated in {t2-t1:.1f}s")

    # Compute normalization statistics from training data only
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std[std < 1e-8] = 1.0  # avoid division by zero for constant features
    target_mean = float(train_targets.mean())
    target_std = float(train_targets.std())

    # Save
    np.savez_compressed(train_path, features=train_features.astype(np.float32),
                        targets=train_targets.astype(np.float32))
    np.savez_compressed(val_path, features=val_features.astype(np.float32),
                        targets=val_targets.astype(np.float32))
    np.savez_compressed(stats_path, mean=mean.astype(np.float32), std=std.astype(np.float32),
                        target_mean=np.float32(target_mean), target_std=np.float32(target_std))

    print(f"\nData saved to {DATA_DIR}")
    print(f"  Training:   {train_path} ({n_train:,} samples)")
    print(f"  Validation: {val_path} ({n_val:,} samples)")
    print(f"  Stats:      {stats_path}")

    # Print data summary
    print(f"\nData statistics:")
    print(f"  TL range:  [{train_targets.min():.1f}, {train_targets.max():.1f}] dB")
    print(f"  TL mean:   {target_mean:.1f} dB")
    print(f"  TL std:    {target_std:.1f} dB")
    print(f"  Freq range: [{train_features[:, IDX_FREQ].min():.0f}, {train_features[:, IDX_FREQ].max():.0f}] Hz")
    print(f"  Range:     [{train_features[:, IDX_RANGE].min():.1f}, {train_features[:, IDX_RANGE].max():.1f}] km")


# ---------------------------------------------------------------------------
# Runtime utilities (imported by train.py)
# ---------------------------------------------------------------------------

@dataclass
class FeatureStats:
    """Normalization statistics computed from training data.
    Replaces the Tokenizer role from the original autoresearch.
    """
    mean: torch.Tensor       # (MAX_FEATURES,)
    std: torch.Tensor        # (MAX_FEATURES,)
    target_mean: float
    target_std: float

    @classmethod
    def from_data(cls, data_dir=DATA_DIR):
        stats_path = os.path.join(data_dir, "stats.npz")
        data = np.load(stats_path)
        return cls(
            mean=torch.from_numpy(data["mean"]),
            std=torch.from_numpy(data["std"]),
            target_mean=float(data["target_mean"]),
            target_std=float(data["target_std"]),
        )

    def normalize(self, x):
        """Normalize input features. x: (..., MAX_FEATURES)"""
        return (x - self.mean.to(x.device)) / self.std.to(x.device)

    def denormalize_target(self, y):
        """Convert normalized targets back to dB."""
        return y * self.target_std + self.target_mean


def make_dataloader(stats, B, split):
    """
    Infinite batched dataloader for TL data.

    Args:
        stats: FeatureStats instance (for normalization)
        B: batch size
        split: "train" or "val"

    Yields:
        (inputs: FloatTensor(B, MAX_FEATURES), targets: FloatTensor(B,), epoch: int)
        - inputs are normalized using stats
        - targets are raw TL values in dB (NOT normalized)
    """
    assert split in ["train", "val"]
    data_path = os.path.join(DATA_DIR, f"{split}.npz")
    data = np.load(data_path)
    features = torch.from_numpy(data["features"])  # (N, MAX_FEATURES)
    targets = torch.from_numpy(data["targets"])     # (N,)
    N = features.shape[0]

    # Normalize features
    features = stats.normalize(features)

    # Pin memory for faster GPU transfer
    features = features.pin_memory()
    targets = targets.pin_memory()

    # Pre-allocate GPU tensors
    gpu_features = torch.empty(B, MAX_FEATURES, dtype=torch.float32, device="cuda")
    gpu_targets = torch.empty(B, dtype=torch.float32, device="cuda")

    epoch = 1
    while True:
        if split == "train":
            perm = torch.randperm(N)
            features = features[perm]
            targets = targets[perm]

        for start in range(0, N - B + 1, B):
            gpu_features.copy_(features[start:start+B], non_blocking=True)
            gpu_targets.copy_(targets[start:start+B], non_blocking=True)
            yield gpu_features, gpu_targets, epoch

        epoch += 1


# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — this is the fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_rmse(model, stats, batch_size):
    """
    Root Mean Square Error (dB) on validation set.
    Lower is better. This is the fixed evaluation metric.

    The model's forward signature must be:
        model(x) -> predictions of shape (B,)
    where x is normalized features of shape (B, MAX_FEATURES).

    Args:
        model: trained model in eval mode
        stats: FeatureStats instance
        batch_size: batch size for evaluation

    Returns:
        float: RMSE in dB on the validation set
    """
    val_loader = make_dataloader(stats, batch_size, "val")
    steps = EVAL_SAMPLES // batch_size
    total_se = 0.0
    total_n = 0

    for _ in range(steps):
        x, y, _ = next(val_loader)
        pred = model(x)  # (B,)
        if pred.dim() > 1:
            pred = pred.squeeze(-1)
        se = ((pred - y) ** 2).sum().item()
        total_se += se
        total_n += y.size(0)

    return math.sqrt(total_se / total_n)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Phase 2: pyram (parabolic equation) data generation
# ---------------------------------------------------------------------------

def run_pyram_scenario(freq, zs, ssp_depths, ssp_values, water_depth,
                       bathy_ranges_km, bathy_depths,
                       bottom_ss, bottom_density, max_range_km):
    """Run pyram for a single scenario, return (vr_km, vz, tlg) or None on failure.

    Args:
        freq: Frequency in Hz
        zs: Source depth in m
        ssp_depths: SSP depth array (m)
        ssp_values: Sound speed values (m/s) at ssp_depths
        water_depth: Water depth at source (m)
        bathy_ranges_km: Bathymetry range points (km), length 20
        bathy_depths: Bathymetry depth values (m), length 20
        bottom_ss: Bottom sound speed (m/s)
        bottom_density: Bottom density (g/cm3)
        max_range_km: Maximum range (km)

    Returns:
        (vr_km, vz, tlg) or None if pyram fails
    """
    from pyram.PyRAM import PyRAM

    max_range_m = max_range_km * 1000.0

    # Build SSP that covers full water column
    # pyram requires SSP to extend to at least the deepest bathymetry point
    max_bathy = float(np.max(bathy_depths))
    required_depth = max(water_depth, max_bathy) + 1.0

    z_ss = ssp_depths.copy()
    cw_1d = ssp_values.copy()

    # Ensure z_ss starts at 0
    if z_ss[0] > 0:
        z_ss = np.concatenate([[0.0], z_ss])
        cw_1d = np.concatenate([cw_1d[:1], cw_1d])

    # Extend SSP to cover full depth if needed
    if z_ss[-1] < required_depth:
        z_ss = np.append(z_ss, required_depth)
        cw_1d = np.append(cw_1d, cw_1d[-1])  # extrapolate with last value

    if len(z_ss) < 2:
        z_ss = np.array([0.0, required_depth])
        cw_1d = np.array([1500.0, 1500.0])

    # Range-independent SSP (single column)
    rp_ss = np.array([0.0])
    cw = cw_1d.reshape(-1, 1)

    # Seabed parameters (range-independent, single point)
    z_sb = np.array([0.0])
    rp_sb = np.array([0.0])
    cb = np.array([[float(bottom_ss)]])
    rhob = np.array([[float(bottom_density)]])
    attn = np.array([[0.5]])  # dB/wavelength

    # Bathymetry: convert to [range_m, depth_m] format
    bathy_r_m = bathy_ranges_km * 1000.0
    rbzb = np.column_stack([bathy_r_m, bathy_depths])
    # Ensure starts at 0 and ends at max_range
    if rbzb[0, 0] > 0:
        rbzb = np.vstack([[0.0, rbzb[0, 1]], rbzb])
    if rbzb[-1, 0] < max_range_m:
        rbzb = np.vstack([rbzb, [max_range_m, rbzb[-1, 1]]])

    try:
        zr = min(zs, water_depth - 1.0)  # dummy receiver depth (we use full grid)
        zr = max(1.0, zr)
        model = PyRAM(freq, zs, zr, z_ss, rp_ss, cw,
                      z_sb, rp_sb, cb, rhob, attn, rbzb,
                      rmax=max_range_m)
        model.run()

        # Filter out invalid TL values
        tlg = model.tlg
        if np.any(np.isnan(tlg)) or np.any(np.isinf(tlg)):
            tlg = np.nan_to_num(tlg, nan=200.0, posinf=200.0, neginf=10.0)

        return model.vr / 1000.0, model.vz, tlg  # vr in km
    except Exception:
        return None


def _generate_pyram_worker(args):
    """Worker function for parallel pyram data generation."""
    scenario_idx, seed, samples_per_run = args
    rng = np.random.default_rng(seed)

    # Sample random environment — constrained to keep pyram runtime < ~2s
    # Grid size ∝ freq * depth * range, so limit the product
    freq = 10 ** rng.uniform(1.3, 3.3)  # 20 Hz - 2 kHz (safe range)

    regime = rng.choice(["shallow", "medium", "deep"], p=[0.35, 0.40, 0.25])
    if regime == "shallow":
        water_depth = rng.uniform(30, 200)
    elif regime == "medium":
        water_depth = rng.uniform(200, 1000)
    else:
        water_depth = rng.uniform(1000, 3000)

    src_depth = rng.uniform(5, min(300, water_depth - 5))

    # Limit range to keep pyram grid manageable: freq*depth*range < budget
    # Budget ~5e8 keeps most runs under 2s
    grid_budget = 5e8
    max_possible_range_km = min(200, grid_budget / (freq * water_depth) / 1000)
    max_possible_range_km = max(2, min(max_possible_range_km, 100))
    max_range_km = rng.uniform(2, max_possible_range_km)

    # Bottom type
    bottom_types = list(BOTTOM_TYPES.keys())
    bt = rng.choice(bottom_types)
    bottom_ss = BOTTOM_TYPES[bt]["soundspeed"] + rng.normal(0, 20)
    bottom_density = BOTTOM_TYPES[bt]["density"] + rng.normal(0, 0.1)
    bottom_density = max(1.0, bottom_density)

    # SSP
    ssp_type = rng.choice(["isovelocity", "thermocline", "deep_channel", "arctic", "shallow"],
                          p=[0.15, 0.30, 0.25, 0.15, 0.15])
    ssp_values = generate_ssp_profile(ssp_type, water_depth, rng)

    # Bathymetry
    bathy_type = rng.choice(["flat", "slope", "shelf_break", "seamount", "rough"],
                            p=[0.25, 0.25, 0.20, 0.15, 0.15])
    if bathy_type == "flat":
        water_depth_rcv = water_depth + rng.normal(0, water_depth * 0.05)
    elif bathy_type == "slope":
        water_depth_rcv = water_depth * rng.uniform(0.3, 3.0)
    elif bathy_type == "shelf_break":
        water_depth_rcv = rng.uniform(1500, 4500) if water_depth < 500 else rng.uniform(50, 300)
    elif bathy_type == "seamount":
        water_depth_rcv = water_depth + rng.normal(0, water_depth * 0.1)
    else:
        water_depth_rcv = water_depth + rng.normal(0, water_depth * 0.15)
    water_depth_rcv = max(20, min(water_depth_rcv, 6000))

    bathy_profile = generate_bathymetry_profile(
        bathy_type, water_depth, water_depth_rcv, 20, rng)

    # Bathymetry range points (20 evenly spaced)
    bathy_ranges_km = np.linspace(0, max_range_km, 20)

    # Run pyram
    result = run_pyram_scenario(
        freq, src_depth, SSP_DEPTHS, ssp_values, water_depth,
        bathy_ranges_km, bathy_profile,
        bottom_ss, bottom_density, max_range_km)

    if result is None:
        return None

    vr_km, vz, tlg = result

    # Sample random (receiver_depth, range) points from the grid
    n_depths, n_ranges = tlg.shape
    if n_depths < 2 or n_ranges < 2:
        return None

    features_list = []
    targets_list = []

    for _ in range(samples_per_run):
        # Random range index (skip very close range)
        ri = rng.integers(max(1, n_ranges // 20), n_ranges)
        # Random depth index (within water column at this range)
        # Estimate water depth at this range from bathymetry
        range_frac = vr_km[ri] / max_range_km if max_range_km > 0 else 0
        local_depth = np.interp(range_frac, np.linspace(0, 1, 20), bathy_profile)
        max_di = np.searchsorted(vz, local_depth) - 1
        max_di = max(1, min(max_di, n_depths - 1))
        di = rng.integers(0, max_di + 1)

        rcv_depth = vz[di]
        range_km = vr_km[ri]
        tl = tlg[di, ri]

        # Bounds check
        if tl < 10 or tl > 200 or np.isnan(tl):
            continue

        # Local water depth at receiver
        wd_rcv = np.interp(range_frac, np.linspace(0, 1, 20), bathy_profile)

        # Pack feature vector
        feat = np.zeros(MAX_FEATURES, dtype=np.float64)
        feat[IDX_FREQ] = freq
        feat[IDX_SRC_DEPTH] = src_depth
        feat[IDX_RCV_DEPTH] = rcv_depth
        feat[IDX_RANGE] = range_km
        feat[IDX_WATER_DEPTH_SRC] = water_depth
        feat[IDX_WATER_DEPTH_RCV] = wd_rcv
        feat[IDX_BOTTOM_SS] = bottom_ss
        feat[IDX_BOTTOM_DENSITY] = bottom_density
        feat[IDX_SSP_START:IDX_SSP_END] = ssp_values
        feat[IDX_BATHY_START:IDX_BATHY_END] = bathy_profile

        features_list.append(feat)
        targets_list.append(tl)

    if len(features_list) == 0:
        return None

    return np.array(features_list), np.array(targets_list)


def generate_phase2_data(n_runs=4000, samples_per_run=250,
                         val_runs=500, val_samples_per_run=100,
                         n_workers=4):
    """Generate Phase 2 data using pyram parabolic equation model.

    Args:
        n_runs: Number of pyram scenarios for training
        samples_per_run: Samples extracted per scenario (training)
        val_runs: Number of pyram scenarios for validation
        val_samples_per_run: Samples extracted per scenario (validation)
        n_workers: Number of parallel workers
    """
    from multiprocessing import Pool

    os.makedirs(DATA_DIR, exist_ok=True)

    train_path = os.path.join(DATA_DIR, "train.npz")
    val_path = os.path.join(DATA_DIR, "val.npz")
    stats_path = os.path.join(DATA_DIR, "stats.npz")

    # Training data
    print(f"Phase 2: Generating training data ({n_runs} pyram runs × {samples_per_run} samples)...")
    t0 = time.time()

    train_args = [(i, 1000000 + i, samples_per_run) for i in range(n_runs)]

    all_features = []
    all_targets = []

    if n_workers > 1:
        with Pool(n_workers) as pool:
            for i, result in enumerate(pool.imap_unordered(_generate_pyram_worker, train_args)):
                if result is not None:
                    all_features.append(result[0])
                    all_targets.append(result[1])
                if (i + 1) % 100 == 0:
                    n_samples = sum(f.shape[0] for f in all_features)
                    elapsed = time.time() - t0
                    print(f"  [{i+1}/{n_runs}] {n_samples:,} samples, {elapsed:.0f}s elapsed")
    else:
        for i, args in enumerate(train_args):
            result = _generate_pyram_worker(args)
            if result is not None:
                all_features.append(result[0])
                all_targets.append(result[1])
            if (i + 1) % 100 == 0:
                n_samples = sum(f.shape[0] for f in all_features)
                elapsed = time.time() - t0
                print(f"  [{i+1}/{n_runs}] {n_samples:,} samples, {elapsed:.0f}s elapsed")

    train_features = np.concatenate(all_features).astype(np.float32)
    train_targets = np.concatenate(all_targets).astype(np.float32)
    t1 = time.time()
    print(f"  Training: {train_features.shape[0]:,} samples in {t1-t0:.0f}s")

    # Validation data (different seeds)
    print(f"Phase 2: Generating validation data ({val_runs} pyram runs)...")
    val_args = [(i, 9000000 + i, val_samples_per_run) for i in range(val_runs)]

    val_features_list = []
    val_targets_list = []

    if n_workers > 1:
        with Pool(n_workers) as pool:
            for result in pool.imap_unordered(_generate_pyram_worker, val_args):
                if result is not None:
                    val_features_list.append(result[0])
                    val_targets_list.append(result[1])
    else:
        for args in val_args:
            result = _generate_pyram_worker(args)
            if result is not None:
                val_features_list.append(result[0])
                val_targets_list.append(result[1])

    val_features = np.concatenate(val_features_list).astype(np.float32)
    val_targets = np.concatenate(val_targets_list).astype(np.float32)
    t2 = time.time()
    print(f"  Validation: {val_features.shape[0]:,} samples in {t2-t1:.0f}s")

    # Compute normalization statistics from training data
    mean = train_features.mean(axis=0)
    std = train_features.std(axis=0)
    std[std < 1e-8] = 1.0
    target_mean = float(train_targets.mean())
    target_std = float(train_targets.std())

    # Save (overwrites Phase 1 data)
    np.savez_compressed(train_path, features=train_features, targets=train_targets)
    np.savez_compressed(val_path, features=val_features, targets=val_targets)
    np.savez_compressed(stats_path, mean=mean.astype(np.float32), std=std.astype(np.float32),
                        target_mean=np.float32(target_mean), target_std=np.float32(target_std))

    print(f"\nPhase 2 data saved to {DATA_DIR}")
    print(f"  Training:   {train_features.shape[0]:,} samples")
    print(f"  Validation: {val_features.shape[0]:,} samples")
    print(f"  TL range:   [{train_targets.min():.1f}, {train_targets.max():.1f}] dB")
    print(f"  TL mean:    {target_mean:.1f} dB")
    print(f"  TL std:     {target_std:.1f} dB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for autoresearch-tl")
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2],
                        help="Data generation phase (1=analytical, 2=pyram)")
    parser.add_argument("--num-train", type=int, default=200_000,
                        help="Number of training samples for Phase 1 (default: 200000)")
    parser.add_argument("--num-val", type=int, default=50_000,
                        help="Number of validation samples for Phase 1 (default: 50000)")
    parser.add_argument("--runs", type=int, default=4000,
                        help="Number of pyram runs for Phase 2 training (default: 4000)")
    parser.add_argument("--samples-per-run", type=int, default=250,
                        help="Samples per pyram run for Phase 2 (default: 250)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers for Phase 2 (default: 4)")
    args = parser.parse_args()

    print(f"Cache directory: {CACHE_DIR}")
    print()

    if args.phase == 1:
        generate_phase1_data(n_train=args.num_train, n_val=args.num_val)
    else:
        generate_phase2_data(n_runs=args.runs, samples_per_run=args.samples_per_run,
                             n_workers=args.workers)
    print()
    print("Done! Ready to train.")
