#!/usr/bin/env python3
"""Create pure VGSSM submission with just prefix fix."""

import pandas as pd
import numpy as np
from pathlib import Path

print("=== Pure VGSSM Submission ===\n")

# Load the baseline VGSSM that scored 0.18
vgssm = pd.read_parquet("submission_vgssm_v1_final.parquet")
print(f"Loaded VGSSM: {len(vgssm):,} rows")

# Check current state
print("\nBefore prefix fix:")
for mid in [1, 2]:
    mdf = vgssm[vgssm["model_id"] == mid]
    wl = mdf["water_level"]
    print(f"  Model {mid}: [{wl.min():.2f}, {wl.max():.2f}], mean={wl.mean():.2f}")

# Add timestep
sample = pd.read_parquet("data/sample_submission.parquet")
sample = sample.sort_values("row_id").reset_index(drop=True)
sample["timestep"] = sample.groupby(["model_id", "event_id", "node_type", "node_id"]).cumcount()

vgssm = vgssm.merge(sample[["row_id", "timestep"]], on="row_id")

# Apply prefix fix
print("\nApplying prefix fix...")
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
vgssm = vgssm.merge(obs_df, on=["model_id", "event_id", "node_type", "node_id", "timestep"], how="left")

mask = vgssm["obs_water_level"].notna()
print(f"Replacing {mask.sum():,} prefix values")
vgssm.loc[mask, "water_level"] = vgssm.loc[mask, "obs_water_level"]

# Clean up
out = vgssm[["row_id", "model_id", "event_id", "node_type", "node_id", "water_level"]].copy()

print("\nAfter prefix fix:")
for mid in [1, 2]:
    mdf = out[out["model_id"] == mid]
    wl = mdf["water_level"]
    print(f"  Model {mid}: [{wl.min():.2f}, {wl.max():.2f}], mean={wl.mean():.2f}")

out.to_parquet("submission_pure_vgssm.parquet", index=False)
print(f"\nSaved: submission_pure_vgssm.parquet")
