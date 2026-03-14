#!/usr/bin/env python3
"""Add observed prefix data to submission."""

import pandas as pd
import numpy as np
from pathlib import Path

print("Loading submission...")
sub = pd.read_parquet("submission_fixed_v3.parquet")
print(f"Submission: {len(sub):,} rows")

print("Loading sample to get timestep structure...")
sample = pd.read_parquet("data/sample_submission.parquet")
sample = sample.sort_values("row_id").reset_index(drop=True)
sample["timestep"] = sample.groupby(["model_id", "event_id", "node_type", "node_id"]).cumcount()

# Merge timestep info to our submission
sub = sub.merge(sample[["row_id", "timestep"]], on="row_id", how="left")
print(f"Timestep range: {sub['timestep'].min()} - {sub['timestep'].max()}")

# Load ALL observed data for prefix (timesteps 0-9)
data_dir = Path("data")
prefix_len = 10
all_obs = []

for model_id in [1, 2]:
    model_dir = data_dir / f"Model_{model_id}" / "test"

    for event_dir in sorted(model_dir.glob("event_*")):
        event_id = int(event_dir.name.split("_")[1])

        # Load 1D data
        f1d = event_dir / "1d_nodes_dynamic_all.csv"
        if f1d.exists():
            df1d = pd.read_csv(f1d)
            df1d = df1d[df1d["timestep"] < prefix_len].copy()
            df1d["model_id"] = model_id
            df1d["event_id"] = event_id
            df1d["node_type"] = 1
            df1d = df1d.rename(columns={"node_idx": "node_id", "water_level": "obs_water_level"})
            all_obs.append(df1d[["model_id", "event_id", "node_type", "node_id", "timestep", "obs_water_level"]])

        # Load 2D data
        f2d = event_dir / "2d_nodes_dynamic_all.csv"
        if f2d.exists():
            df2d = pd.read_csv(f2d)
            df2d = df2d[df2d["timestep"] < prefix_len].copy()
            df2d["model_id"] = model_id
            df2d["event_id"] = event_id
            df2d["node_type"] = 2
            df2d = df2d.rename(columns={"node_idx": "node_id", "water_level": "obs_water_level"})
            all_obs.append(df2d[["model_id", "event_id", "node_type", "node_id", "timestep", "obs_water_level"]])

print(f"Loaded observed data from {len(all_obs)} files")

obs_df = pd.concat(all_obs, ignore_index=True)
print(f"Total observed rows: {len(obs_df):,}")

# Merge observed data into submission
sub = sub.merge(
    obs_df,
    on=["model_id", "event_id", "node_type", "node_id", "timestep"],
    how="left"
)

# Replace water_level with observed where available
mask = sub["obs_water_level"].notna()
print(f"Replacing {mask.sum():,} prefix values with observed data")
sub.loc[mask, "water_level"] = sub.loc[mask, "obs_water_level"]

# Validation
print("=== Final Validation ===")
for model_id in [1, 2]:
    mdf = sub[sub["model_id"] == model_id]
    print(f"Model {model_id}: water_level min={mdf['water_level'].min():.4f}, max={mdf['water_level'].max():.4f}, mean={mdf['water_level'].mean():.4f}")

# Clean up and save
sub = sub.drop(columns=["timestep", "obs_water_level"])
sub.to_parquet("submission_final_timer_v4.parquet", index=False)
print(f"Saved to submission_final_timer_v4.parquet")
print(f"Shape: {sub.shape}")
