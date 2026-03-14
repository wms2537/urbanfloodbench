#!/usr/bin/env python3
"""Apply post-processing enhancements to submissions."""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter1d

print("=== Enhancing Submissions ===\n")

# Load the ensemble submission (our best so far)
sub = pd.read_parquet("submission_ensemble_weighted.parquet")
print(f"Loaded: {len(sub):,} rows")

# Add timestep
sample = pd.read_parquet("data/sample_submission.parquet")
sample = sample.sort_values("row_id").reset_index(drop=True)
sample["timestep"] = sample.groupby(["model_id", "event_id", "node_type", "node_id"]).cumcount()
sub = sub.merge(sample[["row_id", "timestep"]], on="row_id")

print("\n1. TEMPORAL SMOOTHING")
# Apply Gaussian smoothing to each node's time series
def smooth_timeseries(group, sigma=1.0):
    wl = group["water_level"].values
    # Only smooth predictions (t >= 10), keep prefix as-is
    ts = group["timestep"].values
    mask_pred = ts >= 10
    if mask_pred.sum() > 0:
        wl_pred = wl[mask_pred]
        wl_smoothed = gaussian_filter1d(wl_pred, sigma=sigma)
        wl[mask_pred] = wl_smoothed
    group["water_level_smooth"] = wl
    return group

print("  Applying Gaussian smoothing (sigma=1.0)...")
sub = sub.groupby(["model_id", "event_id", "node_type", "node_id"], group_keys=False).apply(smooth_timeseries)

print("\n2. PHYSICAL CONSTRAINTS")
# Ensure non-negative water levels (shouldn't happen, but safety check)
negative_count = (sub["water_level_smooth"] < 0).sum()
print(f"  Negative values: {negative_count}")
sub["water_level_smooth"] = sub["water_level_smooth"].clip(lower=0)

print("\n3. VALIDATION")
for mid in [1, 2]:
    mdf = sub[sub["model_id"] == mid]
    wl_orig = mdf["water_level"]
    wl_smooth = mdf["water_level_smooth"]
    print(f"  Model {mid}:")
    print(f"    Original: [{wl_orig.min():.2f}, {wl_orig.max():.2f}], mean={wl_orig.mean():.2f}")
    print(f"    Smoothed: [{wl_smooth.min():.2f}, {wl_smooth.max():.2f}], mean={wl_smooth.mean():.2f}")
    print(f"    MAE change: {(wl_orig - wl_smooth).abs().mean():.4f}")

# Save smoothed version
out = sub[["row_id", "model_id", "event_id", "node_type", "node_id"]].copy()
out["water_level"] = sub["water_level_smooth"]
out.to_parquet("submission_ensemble_smoothed.parquet", index=False)
print(f"\nSaved: submission_ensemble_smoothed.parquet")

# Also try median ensemble
print("\n\n=== MEDIAN ENSEMBLE ===")
# Load multiple submissions
subs = {}
for name, path in [
    ("vgssm_v1", "submission_vgssm_v1_final.parquet"),
    ("vgssm_final", "submission_vgssm_final.parquet"),
    ("vgssm_final_v2", "submission_vgssm_final_v2.parquet"),
    ("timer_corrected", "submission_timer_bias_corrected.parquet"),
]:
    try:
        df = pd.read_parquet(path)
        subs[name] = df
        print(f"  Loaded {name}")
    except:
        pass

if len(subs) >= 2:
    # Create median ensemble
    base = sample[["row_id", "model_id", "event_id", "node_type", "node_id", "timestep"]].copy()

    for name, df in subs.items():
        base = base.merge(
            df[["row_id", "water_level"]].rename(columns={"water_level": f"wl_{name}"}),
            on="row_id",
            how="left"
        )

    wl_cols = [c for c in base.columns if c.startswith("wl_")]
    base["water_level_median"] = base[wl_cols].median(axis=1)

    # Apply prefix
    data_dir = Path("data")
    prefix_len = 10
    all_obs = []

    for model_id in [1, 2]:
        model_dir = data_dir / f"Model_{model_id}" / "test"
        for event_dir in sorted(model_dir.glob("event_*")):
            event_id = int(event_dir.name.split("_")[1])
            for node_type, node_file in [(1, "1d_nodes_dynamic_all.csv"), (2, "2d_nodes_dynamic_all.csv")]:
                f = event_dir / node_file
                if f.exists():
                    df = pd.read_csv(f)
                    df = df[df["timestep"] < prefix_len].copy()
                    df["model_id"] = model_id
                    df["event_id"] = event_id
                    df["node_type"] = node_type
                    df = df.rename(columns={"node_idx": "node_id", "water_level": "obs_water_level"})
                    all_obs.append(df[["model_id", "event_id", "node_type", "node_id", "timestep", "obs_water_level"]])

    obs_df = pd.concat(all_obs, ignore_index=True)
    base = base.merge(obs_df, on=["model_id", "event_id", "node_type", "node_id", "timestep"], how="left")
    mask = base["obs_water_level"].notna()
    base.loc[mask, "water_level_median"] = base.loc[mask, "obs_water_level"]

    out_median = base[["row_id", "model_id", "event_id", "node_type", "node_id"]].copy()
    out_median["water_level"] = base["water_level_median"]

    print("  Median Ensemble Validation:")
    for mid in [1, 2]:
        mdf = out_median[out_median["model_id"] == mid]
        wl = mdf["water_level"]
        print(f"    Model {mid}: [{wl.min():.2f}, {wl.max():.2f}], mean={wl.mean():.2f}")

    out_median.to_parquet("submission_ensemble_median.parquet", index=False)
    print(f"\nSaved: submission_ensemble_median.parquet")
