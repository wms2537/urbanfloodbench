#!/usr/bin/env python3
"""Investigate Model 2 1D node issues."""

import pandas as pd
import numpy as np
from pathlib import Path

print("=== MODEL 2 1D NODE INVESTIGATION ===\n")

# Load submissions
timer = pd.read_parquet("submission_final_timer_v4.parquet")
vgssm = pd.read_parquet("submission_vgssm_v1_final.parquet")

# Load sample for timestep
sample = pd.read_parquet("data/sample_submission.parquet")
sample = sample.sort_values("row_id").reset_index(drop=True)
sample["timestep"] = sample.groupby(["model_id", "event_id", "node_type", "node_id"]).cumcount()

timer = timer.merge(sample[["row_id", "timestep"]], on="row_id")
vgssm = vgssm.merge(sample[["row_id", "timestep"]], on="row_id")

# Focus on Model 2 1D nodes
timer_m2_1d = timer[(timer["model_id"] == 2) & (timer["node_type"] == 1)]
vgssm_m2_1d = vgssm[(vgssm["model_id"] == 2) & (vgssm["node_type"] == 1)]

print(f"Model 2 1D rows: {len(timer_m2_1d):,}")

# Merge for comparison
merged = timer_m2_1d[["row_id", "event_id", "node_id", "timestep", "water_level"]].copy()
merged = merged.rename(columns={"water_level": "timer_wl"})
merged = merged.merge(vgssm_m2_1d[["row_id", "water_level"]], on="row_id")
merged = merged.rename(columns={"water_level": "vgssm_wl"})
merged["diff"] = merged["timer_wl"] - merged["vgssm_wl"]

print("\n1. STATS")
print(f"   Timer range: [{merged['timer_wl'].min():.2f}, {merged['timer_wl'].max():.2f}]")
print(f"   VGSSM range: [{merged['vgssm_wl'].min():.2f}, {merged['vgssm_wl'].max():.2f}]")
print(f"   Diff mean: {merged['diff'].mean():.4f}")
print(f"   Diff std: {merged['diff'].std():.4f}")

print("\n2. BY NODE")
for node_id in sorted(merged["node_id"].unique()):
    ndf = merged[merged["node_id"] == node_id]
    rmse = np.sqrt((ndf["diff"] ** 2).mean())
    bias = ndf["diff"].mean()
    print(f"   Node {node_id:3d}: RMSE={rmse:.4f}, bias={bias:.4f}, timer=[{ndf['timer_wl'].min():.2f},{ndf['timer_wl'].max():.2f}], vgssm=[{ndf['vgssm_wl'].min():.2f},{ndf['vgssm_wl'].max():.2f}]")

print("\n3. SAMPLE PREDICTIONS (node 2, event 1)")
sample_data = merged[(merged["node_id"] == 2) & (merged["event_id"] == 1)].sort_values("timestep")
print("   t | timer | vgssm | diff")
for _, row in sample_data.head(20).iterrows():
    print(f"   {int(row['timestep']):3d} | {row['timer_wl']:.2f} | {row['vgssm_wl']:.2f} | {row['diff']:.2f}")

# Load ground truth to see what the actual values should be
print("\n4. CHECKING GROUND TRUTH FOR MODEL 2 TEST DATA")
data_dir = Path("data/Model_2/test")
event_dirs = sorted(data_dir.glob("event_*"))[:3]  # First 3 events

for event_dir in event_dirs:
    event_id = int(event_dir.name.split("_")[1])
    f1d = event_dir / "1d_nodes_dynamic_all.csv"
    if f1d.exists():
        df = pd.read_csv(f1d)
        wl = df["water_level"]
        print(f"   Event {event_id}: ground truth range [{wl.min():.2f}, {wl.max():.2f}], mean={wl.mean():.2f}")

        # Compare with predictions for prefix
        prefix_gt = df[df["timestep"] < 10]["water_level"]
        timer_prefix = timer_m2_1d[(timer_m2_1d["event_id"] == event_id) & (timer_m2_1d["timestep"] < 10)]["water_level"]
        vgssm_prefix = vgssm_m2_1d[(vgssm_m2_1d["event_id"] == event_id) & (vgssm_m2_1d["timestep"] < 10)]["water_level"]

        if len(prefix_gt) > 0 and len(timer_prefix) > 0:
            print(f"       Prefix GT mean: {prefix_gt.mean():.2f}, Timer: {timer_prefix.mean():.2f}, VGSSM: {vgssm_prefix.mean():.2f}")
