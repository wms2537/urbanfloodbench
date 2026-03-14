#!/usr/bin/env python3
"""Create Timer V5 hybrid submission."""

import pandas as pd
import numpy as np
from pathlib import Path

print("=== Creating Timer V5 Submission ===")

# Load Model 2 Timer V5 predictions
m2_pred = pd.read_parquet("submission_vgssm_physics_model2.parquet")
m2_pred = m2_pred.rename(columns={"node_idx": "node_id"})
m2_pred["node_type"] = m2_pred["node_type"].map({"1d": 1, "2d": 2})
print(f"Model 2 Timer V5: {len(m2_pred):,} rows, water [{m2_pred['water_level'].min():.2f}, {m2_pred['water_level'].max():.2f}]")
print(f"  Timestep range: {m2_pred['timestep'].min()} - {m2_pred['timestep'].max()}")

# Load Model 1 from baseline VGSSM
m1_vgssm = pd.read_parquet("submission_vgssm_v1_final.parquet")
m1 = m1_vgssm[m1_vgssm["model_id"] == 1].copy()
print(f"Model 1 VGSSM: {len(m1):,} rows")

# Load sample for structure
sample = pd.read_parquet("data/sample_submission.parquet")
sample = sample.sort_values("row_id").reset_index(drop=True)
sample["timestep"] = sample.groupby(["model_id", "event_id", "node_type", "node_id"]).cumcount()

# Get Model 2 structure from sample
sample_m2 = sample[sample["model_id"] == 2][["row_id", "model_id", "event_id", "node_type", "node_id", "timestep"]].copy()

# Merge Timer V5 predictions with sample structure
m2 = sample_m2.merge(
    m2_pred[["model_id", "event_id", "node_type", "node_id", "timestep", "water_level"]],
    on=["model_id", "event_id", "node_type", "node_id", "timestep"],
    how="left"
)
print(f"Model 2 after merge: {len(m2):,} rows, {m2['water_level'].isna().sum():,} missing")

# Add row_id and timestep to Model 1
sample_m1 = sample[sample["model_id"] == 1][["row_id", "timestep"]].copy()
m1 = m1.merge(sample_m1, on="row_id", how="left")

# Combine
combined = pd.concat([
    m1[["row_id", "model_id", "event_id", "node_type", "node_id", "timestep", "water_level"]],
    m2[["row_id", "model_id", "event_id", "node_type", "node_id", "timestep", "water_level"]]
], ignore_index=True)
combined = combined.sort_values("row_id").reset_index(drop=True)
print(f"Combined: {len(combined):,} rows")

# Apply prefix fix (this will fill in the missing prefix values)
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
combined = combined.merge(obs_df, on=["model_id", "event_id", "node_type", "node_id", "timestep"], how="left")

# Replace prefix with observed data
mask = combined["obs_water_level"].notna()
print(f"Replacing {mask.sum():,} prefix values")
combined.loc[mask, "water_level"] = combined.loc[mask, "obs_water_level"]

# Check for any remaining NaN
remaining_nan = combined["water_level"].isna().sum()
if remaining_nan > 0:
    print(f"Warning: {remaining_nan:,} remaining NaN values")
    # These shouldn't exist if prefix fix worked correctly
    combined["water_level"] = combined["water_level"].fillna(0)

# Save
out = combined[["row_id", "model_id", "event_id", "node_type", "node_id", "water_level"]].copy()
print(f"\nFinal validation:")
for mid in [1, 2]:
    mdf = out[out["model_id"] == mid]
    print(f"  Model {mid}: [{mdf['water_level'].min():.2f}, {mdf['water_level'].max():.2f}], mean={mdf['water_level'].mean():.2f}")

out.to_parquet("submission_timer_v5_hybrid.parquet", index=False)
print(f"\nSaved: submission_timer_v5_hybrid.parquet")
