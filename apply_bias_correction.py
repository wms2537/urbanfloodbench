#!/usr/bin/env python3
"""Apply temporal bias correction to Timer V4 predictions."""

import pandas as pd
import numpy as np
from pathlib import Path

print("=== Applying Bias Correction to Timer V4 ===\n")

# Load Timer V4 and baseline VGSSM
timer = pd.read_parquet("submission_final_timer_v4.parquet")
vgssm = pd.read_parquet("submission_vgssm_v1_final.parquet")

# Load sample for timestep
sample = pd.read_parquet("data/sample_submission.parquet")
sample = sample.sort_values("row_id").reset_index(drop=True)
sample["timestep"] = sample.groupby(["model_id", "event_id", "node_type", "node_id"]).cumcount()

timer = timer.merge(sample[["row_id", "timestep"]], on="row_id")
vgssm = vgssm.merge(sample[["row_id", "timestep"]], on="row_id")

# Compute bias per (model, node_type, timestep_bucket)
print("Computing bias correction factors...")
merged = timer[["row_id", "model_id", "node_type", "timestep", "water_level"]].copy()
merged = merged.rename(columns={"water_level": "timer_wl"})
merged = merged.merge(vgssm[["row_id", "water_level"]], on="row_id")
merged = merged.rename(columns={"water_level": "vgssm_wl"})
merged["diff"] = merged["timer_wl"] - merged["vgssm_wl"]

# Compute bias by model, node_type, and timestep bucket
merged["ts_bucket"] = (merged["timestep"] // 25) * 25  # 25-step buckets

bias_table = merged.groupby(["model_id", "node_type", "ts_bucket"])["diff"].mean().reset_index()
bias_table = bias_table.rename(columns={"diff": "bias"})

print("Bias correction table:")
print(bias_table.to_string(index=False))

# Apply correction to timer predictions
timer["ts_bucket"] = (timer["timestep"] // 25) * 25
timer = timer.merge(bias_table, on=["model_id", "node_type", "ts_bucket"], how="left")
timer["bias"] = timer["bias"].fillna(0)

# Subtract bias to correct
timer["water_level_corrected"] = timer["water_level"] - timer["bias"]

print("\n=== Validation Before/After Correction ===")
for mid in [1, 2]:
    for nt in [1, 2]:
        subset = timer[(timer["model_id"] == mid) & (timer["node_type"] == nt)]
        if len(subset) > 0:
            before = subset["water_level"]
            after = subset["water_level_corrected"]
            nt_name = "1D" if nt == 1 else "2D"
            print(f"Model {mid} {nt_name}: before mean={before.mean():.2f}, after mean={after.mean():.2f}")

# Add prefix observed data
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
timer = timer.merge(obs_df, on=["model_id", "event_id", "node_type", "node_id", "timestep"], how="left")

mask = timer["obs_water_level"].notna()
print(f"Replacing {mask.sum():,} prefix values")
timer.loc[mask, "water_level_corrected"] = timer.loc[mask, "obs_water_level"]

# Save
out = timer[["row_id", "model_id", "event_id", "node_type", "node_id"]].copy()
out["water_level"] = timer["water_level_corrected"]

print("\n=== Final Validation ===")
for mid in [1, 2]:
    mdf = out[out["model_id"] == mid]
    wl = mdf["water_level"]
    print(f"Model {mid}: [{wl.min():.2f}, {wl.max():.2f}], mean={wl.mean():.2f}")

out.to_parquet("submission_timer_bias_corrected.parquet", index=False)
print(f"\nSaved: submission_timer_bias_corrected.parquet")
