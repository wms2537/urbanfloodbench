#!/usr/bin/env python3
"""Diagnose why Timer V4 underperforms despite good val_rmse."""

import pandas as pd
import numpy as np
import os

print("=== DIAGNOSTIC: Timer V4 vs Best VGSSM (0.18) ===\n")

# Load Timer V4 submission
timer = pd.read_parquet("submission_final_timer_v4.parquet")

# Find best VGSSM submission
vgssm_files = [f for f in os.listdir(".") if "vgssm" in f and f.endswith(".parquet") and "timer" not in f]
print(f"Available VGSSM submissions: {vgssm_files[:5]}")

# Try to load a baseline VGSSM
best_vgssm = None
for f in ["submission_vgssm_v1_final.parquet", "submission_vgssm_final.parquet", "submission_vgssm_v1_fixed.parquet"]:
    if os.path.exists(f):
        best_vgssm = pd.read_parquet(f)
        print(f"Loaded baseline: {f}")
        break

# Load sample for timestep
sample = pd.read_parquet("data/sample_submission.parquet")
sample = sample.sort_values("row_id").reset_index(drop=True)
sample["timestep"] = sample.groupby(["model_id", "event_id", "node_type", "node_id"]).cumcount()

timer = timer.merge(sample[["row_id", "timestep"]], on="row_id")

print("\n1. TIMER V4 STATS")
for mid in [1, 2]:
    mdf = timer[timer["model_id"] == mid]
    wl = mdf["water_level"]
    print(f"   Model {mid}: water [{wl.min():.2f}, {wl.max():.2f}], mean={wl.mean():.2f}")

print("\n2. TEMPORAL VARIANCE (smoothness check)")
for mid in [1, 2]:
    mdf = timer[timer["model_id"] == mid]
    temporal_std = mdf.groupby(["event_id", "node_type", "node_id"])["water_level"].std().mean()
    print(f"   Model {mid}: avg temporal std per node = {temporal_std:.4f}")

print("\n3. BY TIMESTEP BUCKETS")
for t_start in [0, 10, 50, 100, 200, 300]:
    t_end = t_start + 10
    tdf = timer[(timer["timestep"] >= t_start) & (timer["timestep"] < t_end)]
    if len(tdf) > 0:
        wl = tdf["water_level"]
        print(f"   t=[{t_start:3d},{t_end:3d}): mean={wl.mean():.2f}, std={wl.std():.2f}")

print("\n4. PREFIX vs PREDICTION comparison")
prefix = timer[timer["timestep"] < 10]
pred = timer[timer["timestep"] >= 10]
print(f"   Prefix (0-9):  mean={prefix['water_level'].mean():.2f}, std={prefix['water_level'].std():.2f}")
print(f"   Pred (10+):    mean={pred['water_level'].mean():.2f}, std={pred['water_level'].std():.2f}")

# Check discontinuity at t=9->10
print("\n5. DISCONTINUITY CHECK at t=9 -> t=10")
for mid in [1, 2]:
    mdf = timer[timer["model_id"] == mid]
    t9 = mdf[mdf["timestep"] == 9]["water_level"].mean()
    t10 = mdf[mdf["timestep"] == 10]["water_level"].mean()
    print(f"   Model {mid}: t=9 mean={t9:.2f}, t=10 mean={t10:.2f}, jump={t10-t9:.2f}")

# Compare with baseline VGSSM if available
if best_vgssm is not None:
    print("\n6. COMPARISON WITH BASELINE VGSSM")
    best_vgssm = best_vgssm.merge(sample[["row_id", "timestep"]], on="row_id")

    # Merge for comparison
    merged = timer[["row_id", "model_id", "timestep", "water_level"]].copy()
    merged = merged.rename(columns={"water_level": "timer_wl"})
    merged = merged.merge(best_vgssm[["row_id", "water_level"]], on="row_id")
    merged = merged.rename(columns={"water_level": "vgssm_wl"})
    merged["diff"] = merged["timer_wl"] - merged["vgssm_wl"]
    merged["abs_diff"] = merged["diff"].abs()

    print(f"   Overall MAE: {merged['abs_diff'].mean():.4f}")
    print(f"   Correlation: {merged['timer_wl'].corr(merged['vgssm_wl']):.4f}")

    for mid in [1, 2]:
        mdf = merged[merged["model_id"] == mid]
        print(f"   Model {mid}: MAE={mdf['abs_diff'].mean():.4f}, corr={mdf['timer_wl'].corr(mdf['vgssm_wl']):.4f}")

    # Check temporal variance comparison
    print("\n7. TEMPORAL VARIANCE COMPARISON")
    for mid in [1, 2]:
        mdf = merged[merged["model_id"] == mid]
        timer_var = mdf.groupby("timestep")["timer_wl"].std().mean()
        vgssm_var = mdf.groupby("timestep")["vgssm_wl"].std().mean()
        print(f"   Model {mid}: Timer cross-node std={timer_var:.4f}, VGSSM cross-node std={vgssm_var:.4f}")

print("\n8. CHECK FOR CONSTANT/COLLAPSED PREDICTIONS")
for mid in [1, 2]:
    mdf = timer[timer["model_id"] == mid]
    # Check variance per event
    event_vars = mdf.groupby("event_id")["water_level"].var()
    print(f"   Model {mid}: min event var={event_vars.min():.6f}, max event var={event_vars.max():.6f}")

    # Check if predictions are too similar across nodes
    node_means = mdf.groupby(["event_id", "node_type", "node_id"])["water_level"].mean()
    print(f"   Model {mid}: node mean range [{node_means.min():.2f}, {node_means.max():.2f}]")
