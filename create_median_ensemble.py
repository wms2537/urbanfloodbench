#!/usr/bin/env python3
"""Create median ensemble - memory efficient."""

import pandas as pd
import numpy as np
from pathlib import Path
import gc

print("=== Creating Median Ensemble ===\n")

# Load sample structure
sample = pd.read_parquet("data/sample_submission.parquet")
sample = sample.sort_values("row_id").reset_index(drop=True)
sample["timestep"] = sample.groupby(["model_id", "event_id", "node_type", "node_id"]).cumcount()

# Initialize with just row_id
result = sample[["row_id", "model_id", "event_id", "node_type", "node_id", "timestep"]].copy()

# Load submissions one at a time and stack water levels
wl_arrays = []

submissions = [
    "submission_vgssm_v1_final.parquet",
    "submission_vgssm_final.parquet",
    "submission_vgssm_final_v2.parquet",
    "submission_timer_bias_corrected.parquet",
]

for path in submissions:
    try:
        print(f"Loading {path}...")
        df = pd.read_parquet(path)
        # Sort by row_id to ensure alignment
        df = df.sort_values("row_id")
        wl_arrays.append(df["water_level"].values)
        del df
        gc.collect()
    except Exception as e:
        print(f"  Failed: {e}")

print(f"\nLoaded {len(wl_arrays)} submissions")

# Stack and compute median
print("Computing median...")
wl_stack = np.stack(wl_arrays, axis=1)  # [n_rows, n_models]
result["water_level"] = np.median(wl_stack, axis=1)

del wl_stack, wl_arrays
gc.collect()

# Apply prefix
print("\nAdding prefix data...")
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
result = result.merge(obs_df, on=["model_id", "event_id", "node_type", "node_id", "timestep"], how="left")

mask = result["obs_water_level"].notna()
print(f"Replacing {mask.sum():,} prefix values")
result.loc[mask, "water_level"] = result.loc[mask, "obs_water_level"]

# Clean up and save
out = result[["row_id", "model_id", "event_id", "node_type", "node_id", "water_level"]].copy()

print("\n=== Validation ===")
for mid in [1, 2]:
    mdf = out[out["model_id"] == mid]
    wl = mdf["water_level"]
    print(f"Model {mid}: [{wl.min():.2f}, {wl.max():.2f}], mean={wl.mean():.2f}")

out.to_parquet("submission_median_ensemble.parquet", index=False)
print(f"\nSaved: submission_median_ensemble.parquet")
