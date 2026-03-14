#!/usr/bin/env python3
"""Check if bias varies with timestep - test if bidirectional attention is the cause."""

import pandas as pd
import numpy as np

print("=== TEMPORAL BIAS ANALYSIS ===\n")

# Load submissions
timer = pd.read_parquet("submission_final_timer_v4.parquet")
vgssm = pd.read_parquet("submission_vgssm_v1_final.parquet")

# Load sample for timestep
sample = pd.read_parquet("data/sample_submission.parquet")
sample = sample.sort_values("row_id").reset_index(drop=True)
sample["timestep"] = sample.groupby(["model_id", "event_id", "node_type", "node_id"]).cumcount()

timer = timer.merge(sample[["row_id", "timestep"]], on="row_id")
vgssm = vgssm.merge(sample[["row_id", "timestep"]], on="row_id")

# Focus on Model 2 1D nodes (the problem area)
timer_m2_1d = timer[(timer["model_id"] == 2) & (timer["node_type"] == 1)].copy()
vgssm_m2_1d = vgssm[(vgssm["model_id"] == 2) & (vgssm["node_type"] == 1)].copy()

# Merge
merged = timer_m2_1d[["row_id", "event_id", "node_id", "timestep", "water_level"]].copy()
merged = merged.rename(columns={"water_level": "timer_wl"})
merged = merged.merge(vgssm_m2_1d[["row_id", "water_level"]], on="row_id")
merged = merged.rename(columns={"water_level": "vgssm_wl"})
merged["diff"] = merged["timer_wl"] - merged["vgssm_wl"]

print("1. BIAS BY TIMESTEP BUCKET (Model 2 1D)")
for t_start in [0, 10, 25, 50, 100, 150, 200, 250, 300, 350]:
    t_end = t_start + 25
    tdf = merged[(merged["timestep"] >= t_start) & (merged["timestep"] < t_end)]
    if len(tdf) > 0:
        bias = tdf["diff"].mean()
        std = tdf["diff"].std()
        print(f"   t=[{t_start:3d},{t_end:3d}): bias={bias:+.4f}, std={std:.4f}")

print("\n2. LINEAR REGRESSION: Bias vs Timestep")
# Fit linear regression
from scipy import stats
slope, intercept, r, p, se = stats.linregress(merged["timestep"], merged["diff"])
print(f"   Slope: {slope:.6f} (bias change per timestep)")
print(f"   Intercept: {intercept:.4f} (bias at t=0)")
print(f"   R-squared: {r**2:.4f}")
print(f"   P-value: {p:.2e}")

if slope > 0:
    print("   -> Bias INCREASES with timestep (more positive at later times)")
    print("   -> Interpretation: Timer predicts lower values that GROW with time")
elif slope < 0:
    print("   -> Bias DECREASES with timestep (more negative at later times)")
    print("   -> Interpretation: Timer's underprediction gets WORSE over time")

print("\n3. CHECK ALL MODEL/NODE TYPE COMBINATIONS")
for mid in [1, 2]:
    for nt in [1, 2]:
        subset_t = timer[(timer["model_id"] == mid) & (timer["node_type"] == nt)]
        subset_v = vgssm[(vgssm["model_id"] == mid) & (vgssm["node_type"] == nt)]

        merged_sub = subset_t[["row_id", "timestep", "water_level"]].copy()
        merged_sub = merged_sub.rename(columns={"water_level": "timer_wl"})
        merged_sub = merged_sub.merge(subset_v[["row_id", "water_level"]], on="row_id")
        merged_sub = merged_sub.rename(columns={"water_level": "vgssm_wl"})
        merged_sub["diff"] = merged_sub["timer_wl"] - merged_sub["vgssm_wl"]

        bias = merged_sub["diff"].mean()
        nt_name = "1D" if nt == 1 else "2D"
        print(f"   Model {mid} {nt_name}: bias={bias:+.4f}")

print("\n4. HYPOTHESIS TEST: Is Timer averaging over prefix?")
# If Timer uses mean pooling over prefix (t=0-9), it would average the water levels
# Let's check if Timer predictions correlate more with MEAN(prefix) than LAST(prefix)
print("   (Computing mean vs last prefix correlation...)")

# For each (event, node), compute:
# - mean of prefix (t=0-9)
# - last of prefix (t=9)
# - first prediction (t=10)

timer_m2_1d_wide = timer_m2_1d.pivot(index=["event_id", "node_id"], columns="timestep", values="water_level")
vgssm_m2_1d_wide = vgssm_m2_1d.pivot(index=["event_id", "node_id"], columns="timestep", values="water_level")

if 10 in timer_m2_1d_wide.columns:
    prefix_cols = [c for c in range(10) if c in timer_m2_1d_wide.columns]
    if prefix_cols:
        timer_mean_prefix = timer_m2_1d_wide[prefix_cols].mean(axis=1)
        timer_last_prefix = timer_m2_1d_wide[9] if 9 in timer_m2_1d_wide.columns else timer_m2_1d_wide[prefix_cols[-1]]
        timer_first_pred = timer_m2_1d_wide[10]

        vgssm_mean_prefix = vgssm_m2_1d_wide[prefix_cols].mean(axis=1)
        vgssm_last_prefix = vgssm_m2_1d_wide[9] if 9 in vgssm_m2_1d_wide.columns else vgssm_m2_1d_wide[prefix_cols[-1]]
        vgssm_first_pred = vgssm_m2_1d_wide[10]

        print(f"   Timer: corr(mean_prefix, pred@10) = {timer_mean_prefix.corr(timer_first_pred):.4f}")
        print(f"   Timer: corr(last_prefix, pred@10) = {timer_last_prefix.corr(timer_first_pred):.4f}")
        print(f"   VGSSM: corr(mean_prefix, pred@10) = {vgssm_mean_prefix.corr(vgssm_first_pred):.4f}")
        print(f"   VGSSM: corr(last_prefix, pred@10) = {vgssm_last_prefix.corr(vgssm_first_pred):.4f}")
