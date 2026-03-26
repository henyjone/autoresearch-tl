"""
Global ocean data pipeline for TL prediction training.

Downloads and processes real oceanographic data:
- WOA23: Temperature + Salinity → Sound Speed Profile (SSP)
- GEBCO: Global bathymetry
- Sediment DB: Bottom type (future)

Generates training scenarios at real global locations using
multiple acoustic models: PyRAM (PE), Bellhop (ray), KRAKEN (normal modes).

Usage:
    python global_data.py                    # generate training data
    python global_data.py --num-locations 100  # smaller test run
"""

import os
import sys
import math
import time
import argparse
import warnings
from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch-tl")
OCEAN_DIR = os.path.join(CACHE_DIR, "ocean_data")
DATA_DIR = os.path.join(CACHE_DIR, "data")

WOA_TEMP_FILE = os.path.join(OCEAN_DIR, "woa23_temperature_annual_025.nc")
WOA_SAL_FILE = os.path.join(OCEAN_DIR, "woa23_salinity_annual_025.nc")

# ---------------------------------------------------------------------------
# Sound speed from Temperature + Salinity + Depth (Mackenzie 1981)
# ---------------------------------------------------------------------------

def mackenzie_soundspeed(T, S, D):
    """Mackenzie (1981) equation for sound speed in seawater.

    Args:
        T: Temperature (°C), valid 2-30°C
        S: Salinity (PSU), valid 25-40 PSU
        D: Depth (m), valid 0-8000m

    Returns:
        Sound speed (m/s)
    """
    c = (1448.96 + 4.591 * T - 5.304e-2 * T**2 + 2.374e-4 * T**3
         + 1.340 * (S - 35) + 1.630e-2 * D + 1.675e-7 * D**2
         - 1.025e-2 * T * (S - 35) - 7.139e-13 * T * D**3)
    return c


# ---------------------------------------------------------------------------
# WOA23 data loader
# ---------------------------------------------------------------------------

class WOA23:
    """World Ocean Atlas 2023 data interface."""

    # Standard depth levels for SSP extraction (20 levels, matching prepare.py)
    SSP_DEPTHS = np.array([
        0, 25, 50, 75, 100, 150, 200, 300, 500, 700,
        1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500
    ], dtype=np.float64)

    def __init__(self):
        import netCDF4 as nc
        print("Loading WOA23 temperature...")
        self.ds_temp = nc.Dataset(WOA_TEMP_FILE)
        self.temp = self.ds_temp.variables['t_an']  # (1, 102, 720, 1440)
        self.lat = self.ds_temp.variables['lat'][:]
        self.lon = self.ds_temp.variables['lon'][:]
        self.depth = self.ds_temp.variables['depth'][:]

        self.sal_data = None
        if os.path.exists(WOA_SAL_FILE):
            print("Loading WOA23 salinity...")
            self.ds_sal = nc.Dataset(WOA_SAL_FILE)
            self.sal_data = self.ds_sal.variables['s_an']
        else:
            print("WARNING: Salinity file not found, using default S=35 PSU")

        print(f"  Grid: {len(self.lat)}x{len(self.lon)}, {len(self.depth)} depth levels")
        print(f"  Lat range: [{self.lat.min():.1f}, {self.lat.max():.1f}]")
        print(f"  Lon range: [{self.lon.min():.1f}, {self.lon.max():.1f}]")

    def get_indices(self, lat, lon):
        """Get nearest grid indices for a given lat/lon."""
        lat_idx = int(np.argmin(np.abs(self.lat - lat)))
        lon_idx = int(np.argmin(np.abs(self.lon - lon)))
        return lat_idx, lon_idx

    def get_ssp(self, lat, lon):
        """Get Sound Speed Profile at given location.

        Returns:
            ssp_depths: array of depth levels (m)
            ssp_values: array of sound speeds (m/s)
            water_depth: estimated water depth (m, based on deepest valid data)
        """
        lat_idx, lon_idx = self.get_indices(lat, lon)

        # Extract temperature profile
        temp_profile = self.temp[0, :, lat_idx, lon_idx]

        # Extract salinity profile (or use default)
        if self.sal_data is not None:
            sal_profile = self.sal_data[0, :, lat_idx, lon_idx]
        else:
            sal_profile = np.full_like(temp_profile, 35.0)

        # Find valid (non-masked) depth range
        if hasattr(temp_profile, 'mask'):
            valid = ~temp_profile.mask
        else:
            valid = ~np.isnan(temp_profile)

        if not valid.any():
            return None, None, None  # Land point

        # Compute sound speed at all valid depths
        valid_depths = np.array(self.depth[valid], dtype=np.float64)
        valid_temp = np.array(temp_profile[valid], dtype=np.float64)
        if hasattr(sal_profile, '__getitem__'):
            valid_sal = np.array(sal_profile[valid], dtype=np.float64)
        else:
            valid_sal = np.full(int(valid.sum()), 35.0)

        # Handle masked salinity
        if hasattr(valid_sal, 'mask'):
            valid_sal = np.where(valid_sal.mask, 35.0, valid_sal.data)

        ssp_full = mackenzie_soundspeed(valid_temp, valid_sal, valid_depths)

        # Water depth = deepest valid data point
        water_depth = float(valid_depths.max())

        # Interpolate to standard SSP_DEPTHS (20 levels)
        # Only use depths within the water column
        target_depths = np.array(self.SSP_DEPTHS[self.SSP_DEPTHS <= water_depth], dtype=np.float64)
        if len(target_depths) < 3:
            return None, None, None  # Too shallow

        ssp_full = np.asarray(ssp_full, dtype=np.float64).ravel()
        valid_depths = np.asarray(valid_depths, dtype=np.float64).ravel()
        ssp_values = np.interp(target_depths, valid_depths, ssp_full)

        # Pad to 20 levels if water is shallower than 5500m
        ssp_20 = np.zeros(20, dtype=np.float64)
        ssp_20[:len(ssp_values)] = ssp_values
        # Fill remaining with last valid value (constant extrapolation)
        if len(ssp_values) < 20:
            ssp_20[len(ssp_values):] = ssp_values[-1]

        return self.SSP_DEPTHS, ssp_20, water_depth

    def is_ocean(self, lat, lon):
        """Check if a point is in the ocean (has valid temperature data)."""
        lat_idx, lon_idx = self.get_indices(lat, lon)
        temp_surface = self.temp[0, 0, lat_idx, lon_idx]
        if hasattr(temp_surface, 'mask'):
            return not temp_surface.mask
        return not np.isnan(temp_surface)


# ---------------------------------------------------------------------------
# Global typical ocean regions for training
# ---------------------------------------------------------------------------

# Representative ocean regions with characteristic acoustic properties
OCEAN_REGIONS = {
    # Deep ocean - convergence zone propagation
    "north_pacific_deep": {"lat": (25, 45), "lon": (150, 180), "desc": "North Pacific deep water"},
    "south_pacific_deep": {"lat": (-45, -25), "lon": (-160, -120), "desc": "South Pacific deep water"},
    "north_atlantic_deep": {"lat": (25, 45), "lon": (-60, -30), "desc": "North Atlantic deep water"},
    "south_atlantic_deep": {"lat": (-40, -20), "lon": (-30, 0), "desc": "South Atlantic deep water"},
    "indian_ocean_deep": {"lat": (-35, -10), "lon": (60, 90), "desc": "Indian Ocean deep water"},

    # Tropical - warm surface, strong thermocline
    "tropical_pacific": {"lat": (-10, 10), "lon": (160, -160), "desc": "Tropical Pacific"},
    "tropical_atlantic": {"lat": (-10, 10), "lon": (-40, -10), "desc": "Tropical Atlantic"},
    "tropical_indian": {"lat": (-10, 10), "lon": (60, 90), "desc": "Tropical Indian Ocean"},

    # Polar/Arctic - cold, isothermal, surface duct
    "arctic": {"lat": (70, 85), "lon": (-30, 30), "desc": "Arctic Ocean"},
    "antarctic": {"lat": (-70, -60), "lon": (-180, 180), "desc": "Southern Ocean"},

    # Shallow seas
    "south_china_sea": {"lat": (5, 22), "lon": (110, 120), "desc": "South China Sea"},
    "east_china_sea": {"lat": (25, 33), "lon": (120, 130), "desc": "East China Sea"},
    "yellow_sea": {"lat": (33, 39), "lon": (120, 126), "desc": "Yellow Sea"},
    "sea_of_japan": {"lat": (35, 45), "lon": (130, 140), "desc": "Sea of Japan"},
    "mediterranean": {"lat": (32, 42), "lon": (0, 30), "desc": "Mediterranean Sea"},
    "north_sea": {"lat": (51, 60), "lon": (-5, 10), "desc": "North Sea"},
    "persian_gulf": {"lat": (24, 30), "lon": (48, 56), "desc": "Persian Gulf"},

    # Continental shelves
    "us_east_coast": {"lat": (30, 42), "lon": (-78, -70), "desc": "US East Coast shelf"},
    "norwegian_shelf": {"lat": (60, 70), "lon": (0, 15), "desc": "Norwegian shelf"},

    # Strategically important - submarine corridors
    "giuk_gap": {"lat": (60, 68), "lon": (-30, -10), "desc": "GIUK Gap"},
    "bering_strait": {"lat": (64, 68), "lon": (-172, -166), "desc": "Bering Strait area"},
    "taiwan_strait": {"lat": (22, 26), "lon": (117, 121), "desc": "Taiwan Strait"},
    "malacca_strait": {"lat": (-2, 6), "lon": (98, 104), "desc": "Malacca Strait"},
}


# ---------------------------------------------------------------------------
# Bottom type estimation (simplified, pending real sediment DB)
# ---------------------------------------------------------------------------

# Hamilton-Bachman bottom properties
BOTTOM_TYPES = {
    "sand":    {"soundspeed": 1650.0, "density": 1.9, "grain_phi": 1.0},
    "silt":    {"soundspeed": 1550.0, "density": 1.7, "grain_phi": 5.0},
    "clay":    {"soundspeed": 1500.0, "density": 1.5, "grain_phi": 7.0},
    "gravel":  {"soundspeed": 1800.0, "density": 2.0, "grain_phi": -1.0},
    "rock":    {"soundspeed": 2500.0, "density": 2.5, "grain_phi": -4.0},
}

def estimate_bottom_type(water_depth, lat, rng):
    """Estimate bottom type based on water depth and latitude (simplified).

    Deep ocean: clay/silt, Shelf: sand/silt, Polar: silt/clay, Coastal: sand/gravel
    """
    if water_depth > 3000:
        # Abyssal plain: mostly clay
        bt = rng.choice(["clay", "silt"], p=[0.7, 0.3])
    elif water_depth > 1000:
        # Continental slope: silt/clay
        bt = rng.choice(["silt", "clay", "sand"], p=[0.5, 0.3, 0.2])
    elif water_depth > 200:
        # Continental shelf: sand/silt
        bt = rng.choice(["sand", "silt", "gravel"], p=[0.5, 0.35, 0.15])
    else:
        # Shallow coastal: sand/gravel
        bt = rng.choice(["sand", "gravel", "silt"], p=[0.5, 0.3, 0.2])

    props = BOTTOM_TYPES[bt]
    # Add some random variation
    ss = props["soundspeed"] * rng.uniform(0.95, 1.05)
    dens = props["density"] * rng.uniform(0.95, 1.05)
    return ss, dens


# ---------------------------------------------------------------------------
# Scenario generation using real ocean data
# ---------------------------------------------------------------------------

def generate_scenario(woa, region_name, region, rng):
    """Generate a single acoustic scenario from real ocean data.

    Args:
        woa: WOA23 instance
        region_name: name of the ocean region
        region: dict with lat/lon bounds
        rng: numpy random generator

    Returns:
        dict with scenario parameters, or None if invalid
    """
    lat_min, lat_max = region["lat"]
    lon_min, lon_max = region["lon"]

    # Handle wrap-around for longitude (e.g., tropical Pacific: 160 to -160)
    if lon_min > lon_max:
        # Crosses the date line
        if rng.random() < 0.5:
            lon = rng.uniform(lon_min, 180)
        else:
            lon = rng.uniform(-180, lon_max)
    else:
        lon = rng.uniform(lon_min, lon_max)

    lat = rng.uniform(lat_min, lat_max)

    # Get SSP from WOA23
    ssp_depths, ssp_values, water_depth = woa.get_ssp(lat, lon)
    if ssp_values is None:
        return None  # Land or too shallow

    if water_depth < 20:
        return None  # Too shallow for meaningful TL

    # Frequency: 20 Hz to 5000 Hz (log-uniform)
    freq = 10 ** rng.uniform(1.3, 3.7)

    # Source depth: 1m to min(500m, water_depth * 0.8)
    max_src_depth = min(500, water_depth * 0.8)
    src_depth = rng.uniform(1, max(2, max_src_depth))

    # Receiver depth
    max_rcv_depth = water_depth * 0.95
    rcv_depth = rng.uniform(1, max(2, max_rcv_depth))

    # Range: depends on water depth and frequency
    # Deep water: up to 300km for CZ propagation
    # Shallow water: up to ~50km
    if water_depth > 1000:
        max_range_km = min(300, 5e10 / (freq * water_depth) / 1000)
        min_range_km = 1
        # Bias toward longer ranges for deep water
        if water_depth > 2000 and freq < 200:
            min_range_km = max(1, min(50, max_range_km * 0.3))
    elif water_depth > 200:
        max_range_km = min(100, 5e9 / (freq * water_depth) / 1000)
        min_range_km = 0.5
    else:
        max_range_km = min(30, 5e8 / (freq * water_depth) / 1000)
        min_range_km = 0.1

    max_range_km = max(max_range_km, min_range_km + 1)
    range_km = 10 ** rng.uniform(np.log10(min_range_km), np.log10(max_range_km))

    # Bottom properties
    bottom_ss, bottom_density = estimate_bottom_type(water_depth, lat, rng)

    # Simple flat bathymetry (from water_depth, slight random variation)
    bathy_profile = np.full(20, water_depth)
    # Add gentle slope
    slope_factor = rng.uniform(-0.1, 0.1)
    bathy_profile += np.linspace(0, slope_factor * water_depth, 20)
    bathy_profile = np.clip(bathy_profile, 10, 10000)

    return {
        "lat": lat,
        "lon": lon,
        "freq": freq,
        "src_depth": src_depth,
        "rcv_depth": rcv_depth,
        "range_km": range_km,
        "water_depth": water_depth,
        "ssp_depths": ssp_depths,
        "ssp_values": ssp_values,
        "bathy_profile": bathy_profile,
        "bottom_ss": bottom_ss,
        "bottom_density": bottom_density,
        "region": region_name,
    }


# ---------------------------------------------------------------------------
# Acoustic model runners
# ---------------------------------------------------------------------------

def run_pyram(scenario):
    """Run PyRAM parabolic equation model.
    Reuses the same API as prepare.py's run_pyram_scenario.
    """
    from prepare import run_pyram_scenario

    ssp_depths = scenario["ssp_depths"]
    ssp_values = scenario["ssp_values"]
    water_depth = scenario["water_depth"]

    # Only use SSP within water column
    valid_mask = ssp_depths <= water_depth
    valid_depths = ssp_depths[valid_mask]
    valid_ssp = ssp_values[valid_mask]

    if len(valid_depths) < 2:
        return None

    bathy_ranges_km = np.linspace(0, scenario["range_km"], 20)

    try:
        result = run_pyram_scenario(
            freq=scenario["freq"],
            zs=scenario["src_depth"],
            ssp_depths=valid_depths,
            ssp_values=valid_ssp,
            water_depth=water_depth,
            bathy_ranges_km=bathy_ranges_km,
            bathy_depths=scenario["bathy_profile"],
            bottom_ss=scenario["bottom_ss"],
            bottom_density=scenario["bottom_density"],
            max_range_km=scenario["range_km"],
        )
        return result
    except Exception:
        return None


def run_bellhop(scenario):
    """Run Bellhop ray tracing model via arlpy."""
    try:
        import arlpy.uwapm as pm
    except ImportError:
        return None

    freq = scenario["freq"]
    water_depth = scenario["water_depth"]
    max_range_m = scenario["range_km"] * 1000

    # Build environment
    ssp_depths = scenario["ssp_depths"]
    ssp_values = scenario["ssp_values"]
    valid_mask = ssp_depths <= water_depth
    valid_depths = ssp_depths[valid_mask]
    valid_ssp = ssp_values[valid_mask]

    if len(valid_depths) < 2:
        return None

    env = pm.create_env2d(
        frequency=freq,
        rx_range=np.linspace(100, max_range_m, min(500, int(max_range_m / 100))),
        rx_depth=np.linspace(1, water_depth * 0.95, min(100, max(10, int(water_depth / 10)))),
        depth=water_depth,
        soundspeed=list(zip(valid_depths, valid_ssp)),
        bottom_soundspeed=scenario["bottom_ss"],
        bottom_density=scenario["bottom_density"] * 1000,  # arlpy uses kg/m3
        tx_depth=scenario["src_depth"],
    )

    try:
        tl_result = pm.compute_transmission_loss(env)
        return tl_result
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Main data generation
# ---------------------------------------------------------------------------

def generate_global_data(num_locations=1000, samples_per_location=100,
                          seed=42, models=None):
    """Generate training data from global ocean locations.

    Args:
        num_locations: number of random locations to sample
        samples_per_location: TL samples per location
        seed: random seed
        models: list of models to use, e.g. ["pyram", "bellhop"]
    """
    if models is None:
        models = ["pyram"]

    rng = np.random.default_rng(seed)
    woa = WOA23()

    # Feature layout (same as prepare.py)
    from prepare import (IDX_FREQ, IDX_SRC_DEPTH, IDX_RCV_DEPTH, IDX_RANGE,
                         IDX_WATER_DEPTH_SRC, IDX_WATER_DEPTH_RCV,
                         IDX_BOTTOM_SS, IDX_BOTTOM_DENSITY,
                         IDX_SSP_START, IDX_SSP_END,
                         IDX_BATHY_START, IDX_BATHY_END,
                         MAX_FEATURES)

    all_features = []
    all_targets = []

    # Weight regions by importance
    region_names = list(OCEAN_REGIONS.keys())
    region_weights = np.ones(len(region_names))
    # Give more weight to deep ocean (CZ propagation) and strategically important areas
    for i, name in enumerate(region_names):
        if "deep" in name:
            region_weights[i] = 3.0
        elif "strait" in name or "gap" in name:
            region_weights[i] = 2.0
        elif "sea" in name:
            region_weights[i] = 1.5
    region_weights /= region_weights.sum()

    t_start = time.time()
    n_scenarios = 0
    n_samples = 0
    n_failed = 0

    for loc_i in range(num_locations):
        # Pick a region
        region_idx = rng.choice(len(region_names), p=region_weights)
        region_name = region_names[region_idx]
        region = OCEAN_REGIONS[region_name]

        # Generate scenario
        scenario = generate_scenario(woa, region_name, region, rng)
        if scenario is None:
            n_failed += 1
            continue

        # Run acoustic model
        for model_name in models:
            if model_name == "pyram":
                result = run_pyram(scenario)
            elif model_name == "bellhop":
                result = run_bellhop(scenario)
            else:
                continue

            if result is None:
                n_failed += 1
                continue

            if model_name == "pyram":
                vr_km, vz, tlg = result

                # Sample points from TL grid
                n_r = len(vr_km)
                n_z = len(vz)
                n_pts = min(samples_per_location, n_r * n_z)

                for _ in range(n_pts):
                    ri = rng.integers(0, n_r)
                    di = rng.integers(0, n_z)

                    # Skip if receiver deeper than local water depth
                    range_frac = vr_km[ri] / scenario["range_km"] if scenario["range_km"] > 0 else 0
                    local_depth = np.interp(range_frac, np.linspace(0, 1, 20), scenario["bathy_profile"])
                    if vz[di] >= local_depth:
                        continue

                    tl = tlg[di, ri]
                    if np.isnan(tl) or tl < 0 or tl > 250:
                        continue

                    # Pack features
                    feat = np.zeros(MAX_FEATURES, dtype=np.float32)
                    feat[IDX_FREQ] = scenario["freq"]
                    feat[IDX_SRC_DEPTH] = scenario["src_depth"]
                    feat[IDX_RCV_DEPTH] = vz[di]
                    feat[IDX_RANGE] = vr_km[ri]
                    feat[IDX_WATER_DEPTH_SRC] = scenario["water_depth"]
                    feat[IDX_WATER_DEPTH_RCV] = local_depth
                    feat[IDX_BOTTOM_SS] = scenario["bottom_ss"]
                    feat[IDX_BOTTOM_DENSITY] = scenario["bottom_density"]
                    feat[IDX_SSP_START:IDX_SSP_END] = scenario["ssp_values"]
                    feat[IDX_BATHY_START:IDX_BATHY_END] = scenario["bathy_profile"]
                    # feat[48] = scenario["lat"]   # TODO: add lat/lon after feature vector update
                    # feat[49] = scenario["lon"]

                    all_features.append(feat)
                    all_targets.append(np.float32(tl))
                    n_samples += 1

            n_scenarios += 1

        # Progress
        if (loc_i + 1) % 10 == 0:
            elapsed = time.time() - t_start
            rate = n_scenarios / elapsed if elapsed > 0 else 0
            print(f"\r  Location {loc_i+1}/{num_locations} | "
                  f"scenarios: {n_scenarios} | samples: {n_samples:,} | "
                  f"failed: {n_failed} | rate: {rate:.1f}/s | "
                  f"elapsed: {elapsed:.0f}s", end="", flush=True)

    print()  # newline

    if n_samples == 0:
        print("ERROR: No samples generated!")
        return

    features = np.array(all_features)
    targets = np.array(all_targets)

    print(f"\nGenerated {n_samples:,} samples from {n_scenarios} scenarios")
    print(f"  TL range: [{targets.min():.1f}, {targets.max():.1f}] dB")
    print(f"  TL mean:  {targets.mean():.1f} dB, std: {targets.std():.1f} dB")

    # Save
    out_path = os.path.join(DATA_DIR, "global_train.npz")
    np.savez_compressed(out_path, features=features, targets=targets)
    print(f"  Saved to: {out_path} ({os.path.getsize(out_path) / 1e6:.1f} MB)")

    return features, targets


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-locations", type=int, default=1000)
    parser.add_argument("--samples-per-location", type=int, default=100)
    parser.add_argument("--models", nargs="+", default=["pyram"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    generate_global_data(
        num_locations=args.num_locations,
        samples_per_location=args.samples_per_location,
        seed=args.seed,
        models=args.models,
    )
