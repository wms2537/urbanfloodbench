#!/usr/bin/env python3
"""Create ensemble submission from multiple models."""

import pandas as pd
import numpy as np
from pathlib import Path

print("=== Creating Ensemble Submission ===\n")

# Load all available submissions
submissions = {}

files_to_try = [
    ("timer_v4", "submission_final_timer_v4.parquet"),
    ("vgssm_v1", "submission_vgssm_v1_final.parquet"),
    ("vgssm_final", "submission_vgssm_final.parquet"),
    ("vgssm_final_v2", "submission_vgssm_final_v2.parquet"),
    ("hybrid", "submission_hybrid_fixed.parquet"),
]

for name, path in files_to_try:
    try:
        df = pd.read_parquet(path)
        submissions[name] = df
        print(f"Loaded {name}: {len(df):,} rows")
    except Exception as e:
        print(f"Failed to load {name}: {e}")

# Load sample for structure
sample = pd.read_parquet("data/sample_submission.parquet")
sample = sample.sort_values("row_id").reset_index(drop=True)

print(f"\nAvailable submissions: {list(submissions.keys())}")

# Strategy 1: Simple average of all models
print("\n=== Strategy 1: Simple Average ===")
base = sample[["row_id", "model_id", "event_id", "node_type", "node_id"]].copy()

# Merge all water levels
for name, df in submissions.items():
    base = base.merge(
        df[["row_id", "water_level"]].rename(columns={"water_level": f"wl_{name}"}),
        on="row_id",
        how="left"
    )

# Average all
wl_cols = [c for c in base.columns if c.startswith("wl_")]
print(f"Averaging {len(wl_cols)} models: {wl_cols}")
base["water_level_avg"] = base[wl_cols].mean(axis=1)

# Strategy 2: Weighted average (more weight to better models)
print("\n=== Strategy 2: Weighted Average ===")
# Based on our analysis, vgssm_v1 is best, timer_v4 is worst for Model 2
weights = {
    "wl_timer_v4": 0.5,
    "wl_vgssm_v1": 2.0,
    "wl_vgssm_final": 1.5,
    "wl_vgssm_final_v2": 1.5,
    "wl_hybrid": 1.0,
}

weighted_sum = sum(base[col] * weights.get(col, 1.0) for col in wl_cols if col in base.columns)
weight_total = sum(weights.get(col, 1.0) for col in wl_cols if col in base.columns)
base["water_level_weighted"] = weighted_sum / weight_total

# Strategy 3: Model-specific ensemble
print("\n=== Strategy 3: Model-Specific Ensemble ===")
# Model 1: Use vgssm models (Timer V4 has small bias, could include)
# Model 2: Exclude Timer V4 (has large bias)

model1_cols = ["wl_vgssm_v1", "wl_vgssm_final", "wl_vgssm_final_v2", "wl_timer_v4"]
model2_cols = ["wl_vgssm_v1", "wl_vgssm_final", "wl_vgssm_final_v2"]  # Exclude timer

model1_cols = [c for c in model1_cols if c in base.columns]
model2_cols = [c for c in model2_cols if c in base.columns]

base["water_level_smart"] = np.where(
    base["model_id"] == 1,
    base[model1_cols].mean(axis=1),
    base[model2_cols].mean(axis=1)
)

# Add timestep for prefix handling
sample["timestep"] = sample.groupby(["model_id", "event_id", "node_type", "node_id"]).cumcount()
base = base.merge(sample[["row_id", "timestep"]], on="row_id")

# Load observed prefix data
print("\n=== Adding Prefix Data ===")
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

# Replace prefix with observed
mask = base["obs_water_level"].notna()
print(f"Replacing {mask.sum():,} prefix values")

for col in ["water_level_avg", "water_level_weighted", "water_level_smart"]:
    base.loc[mask, col] = base.loc[mask, "obs_water_level"]

# Save all strategies
print("\n=== Saving Submissions ===")
for strategy, col in [("avg", "water_level_avg"), ("weighted", "water_level_weighted"), ("smart", "water_level_smart")]:
    out = base[["row_id", "model_id", "event_id", "node_type", "node_id"]].copy()
    out["water_level"] = base[col]

    # Validate
    for mid in [1, 2]:
        mdf = out[out["model_id"] == mid]
        wl = mdf["water_level"]
        print(f"  {strategy} Model {mid}: [{wl.min():.2f}, {wl.max():.2f}], mean={wl.mean():.2f}")

    out.to_parquet(f"submission_ensemble_{strategy}.parquet", index=False)
    print(f"  Saved: submission_ensemble_{strategy}.parquet")
