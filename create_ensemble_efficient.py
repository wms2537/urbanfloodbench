#!/usr/bin/env python3
"""Create ensemble submission - memory efficient version."""

import pandas as pd
import numpy as np
from pathlib import Path
import gc

print("=== Creating Ensemble Submission (Memory Efficient) ===\n")

# Load sample structure
sample = pd.read_parquet("data/sample_submission.parquet")
sample = sample.sort_values("row_id").reset_index(drop=True)
sample["timestep"] = sample.groupby(["model_id", "event_id", "node_type", "node_id"]).cumcount()

# Initialize result with structure
result = sample[["row_id", "model_id", "event_id", "node_type", "node_id", "timestep"]].copy()
result["wl_sum"] = 0.0
result["wl_count"] = 0

# Process submissions one at a time
submissions = [
    ("vgssm_v1", "submission_vgssm_v1_final.parquet", 2.0),  # Best baseline
    ("vgssm_final", "submission_vgssm_final.parquet", 1.5),
    ("vgssm_final_v2", "submission_vgssm_final_v2.parquet", 1.5),
    ("timer_corrected", "submission_timer_bias_corrected.parquet", 1.0),  # Bias-corrected timer
]

for name, path, weight in submissions:
    try:
        print(f"Loading {name}...")
        df = pd.read_parquet(path)

        # Merge water level
        result = result.merge(
            df[["row_id", "water_level"]].rename(columns={"water_level": "wl_temp"}),
            on="row_id",
            how="left"
        )

        # Add to weighted sum (handle Model 2 differently for timer)
        if "timer" in name:
            # Timer: use for Model 1, reduce weight for Model 2
            m1_mask = result["model_id"] == 1
            result.loc[m1_mask, "wl_sum"] += result.loc[m1_mask, "wl_temp"] * weight
            result.loc[m1_mask, "wl_count"] += weight

            # Model 2: lower weight for timer
            m2_mask = result["model_id"] == 2
            result.loc[m2_mask, "wl_sum"] += result.loc[m2_mask, "wl_temp"] * 0.5
            result.loc[m2_mask, "wl_count"] += 0.5
        else:
            result["wl_sum"] += result["wl_temp"].fillna(0) * weight
            result["wl_count"] += weight

        result = result.drop(columns=["wl_temp"])
        del df
        gc.collect()
        print(f"  Processed {name}")

    except Exception as e:
        print(f"  Failed to load {name}: {e}")

# Compute weighted average
result["water_level"] = result["wl_sum"] / result["wl_count"]

# Load and apply prefix data
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
result = result.merge(obs_df, on=["model_id", "event_id", "node_type", "node_id", "timestep"], how="left")

mask = result["obs_water_level"].notna()
print(f"Replacing {mask.sum():,} prefix values")
result.loc[mask, "water_level"] = result.loc[mask, "obs_water_level"]

# Clean up and save
out = result[["row_id", "model_id", "event_id", "node_type", "node_id", "water_level"]].copy()

print("\n=== Final Validation ===")
for mid in [1, 2]:
    mdf = out[out["model_id"] == mid]
    wl = mdf["water_level"]
    print(f"Model {mid}: [{wl.min():.2f}, {wl.max():.2f}], mean={wl.mean():.2f}")

out.to_parquet("submission_ensemble_weighted.parquet", index=False)
print(f"\nSaved: submission_ensemble_weighted.parquet")
