#!/usr/bin/env python3
"""Apply prefix fix to hybrid submission."""

import pandas as pd
import numpy as np
from pathlib import Path

print("=== Fixing Hybrid Submission Prefix ===\n")

# Load hybrid submission
sub = pd.read_parquet("submission_hybrid_v1.parquet")
print(f"Loaded hybrid: {len(sub):,} rows")

# Load sample for timestep
sample = pd.read_parquet("data/sample_submission.parquet")
sample = sample.sort_values("row_id").reset_index(drop=True)
sample["timestep"] = sample.groupby(["model_id", "event_id", "node_type", "node_id"]).cumcount()

sub = sub.merge(sample[["row_id", "timestep"]], on="row_id")

# Load observed data for prefix
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

print(f"Loaded observed data from {len(all_obs)} files")
obs_df = pd.concat(all_obs, ignore_index=True)
print(f"Total observed rows: {len(obs_df):,}")

# Merge observed data
sub = sub.merge(
    obs_df,
    on=["model_id", "event_id", "node_type", "node_id", "timestep"],
    how="left"
)

# Replace with observed where available
mask = sub["obs_water_level"].notna()
print(f"Replacing {mask.sum():,} prefix values")
sub.loc[mask, "water_level"] = sub.loc[mask, "obs_water_level"]

# Clean up
sub = sub.drop(columns=["timestep", "obs_water_level"])

# Validate
print("\n=== Validation ===")
for mid in [1, 2]:
    mdf = sub[sub["model_id"] == mid]
    wl = mdf["water_level"]
    print(f"Model {mid}: water [{wl.min():.2f}, {wl.max():.2f}], mean={wl.mean():.2f}")

# Save
sub.to_parquet("submission_hybrid_fixed.parquet", index=False)
print(f"\nSaved to: submission_hybrid_fixed.parquet")
