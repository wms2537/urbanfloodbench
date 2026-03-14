#!/usr/bin/env python3
"""Deep diagnostic to find what's causing the score difference."""

import pandas as pd
import numpy as np

print("=== DEEP DIAGNOSTIC ===\n")

# Load submissions
timer = pd.read_parquet("submission_final_timer_v4.parquet")
vgssm = pd.read_parquet("submission_vgssm_v1_final.parquet")

# Load sample for timestep
sample = pd.read_parquet("data/sample_submission.parquet")
sample = sample.sort_values("row_id").reset_index(drop=True)
sample["timestep"] = sample.groupby(["model_id", "event_id", "node_type", "node_id"]).cumcount()

timer = timer.merge(sample[["row_id", "timestep"]], on="row_id")
vgssm = vgssm.merge(sample[["row_id", "timestep"]], on="row_id")

# Merge
merged = timer[["row_id", "model_id", "event_id", "node_type", "node_id", "timestep", "water_level"]].copy()
merged = merged.rename(columns={"water_level": "timer_wl"})
merged = merged.merge(vgssm[["row_id", "water_level"]], on="row_id")
merged = merged.rename(columns={"water_level": "vgssm_wl"})
merged["diff"] = merged["timer_wl"] - merged["vgssm_wl"]
merged["sq_diff"] = merged["diff"] ** 2

print("1. ERROR DISTRIBUTION")
print(f"   Mean diff: {merged['diff'].mean():.6f}")
print(f"   Std diff:  {merged['diff'].std():.6f}")
print(f"   Percentiles: 1%={merged['diff'].quantile(0.01):.4f}, 50%={merged['diff'].quantile(0.5):.4f}, 99%={merged['diff'].quantile(0.99):.4f}")

print("\n2. LARGE ERRORS (>2 std)")
large_errors = merged[merged["diff"].abs() > 2 * merged["diff"].std()]
print(f"   Count: {len(large_errors):,} ({100*len(large_errors)/len(merged):.2f}%)")
if len(large_errors) > 0:
    print(f"   By model: {large_errors.groupby('model_id').size().to_dict()}")
    print(f"   By node_type: {large_errors.groupby('node_type').size().to_dict()}")

print("\n3. ERROR BY EVENT")
event_rmse = merged.groupby(["model_id", "event_id"]).apply(
    lambda x: np.sqrt((x["sq_diff"]).mean())
).reset_index(name="rmse")
print(f"   Event RMSE range: [{event_rmse['rmse'].min():.4f}, {event_rmse['rmse'].max():.4f}]")
print(f"   Worst events:")
worst = event_rmse.nlargest(5, "rmse")
print(worst.to_string(index=False))

print("\n4. ERROR BY TIMESTEP (looking for temporal patterns)")
ts_rmse = merged.groupby("timestep").apply(
    lambda x: np.sqrt((x["sq_diff"]).mean())
)
print(f"   Early (t<20) RMSE: {ts_rmse[ts_rmse.index < 20].mean():.4f}")
print(f"   Mid (20-100) RMSE: {ts_rmse[(ts_rmse.index >= 20) & (ts_rmse.index < 100)].mean():.4f}")
print(f"   Late (100+) RMSE:  {ts_rmse[ts_rmse.index >= 100].mean():.4f}")

print("\n5. SYSTEMATIC BIAS CHECK")
for mid in [1, 2]:
    mdf = merged[merged["model_id"] == mid]
    bias = mdf["diff"].mean()
    print(f"   Model {mid}: bias={bias:.4f} (Timer - VGSSM)")

print("\n6. CHECK SPECIFIC HIGH-ERROR REGIONS")
# Look at where Timer differs most from VGSSM
high_diff = merged[merged["diff"].abs() > 1.0]
if len(high_diff) > 0:
    print(f"   Rows with |diff| > 1.0: {len(high_diff):,}")
    # Check temporal pattern of high errors
    ts_dist = high_diff["timestep"].value_counts().sort_index()
    print(f"   High error by timestep (first 20): {ts_dist.head(20).to_dict()}")

print("\n7. NODE-LEVEL ANALYSIS")
node_rmse = merged.groupby(["model_id", "node_type", "node_id"]).apply(
    lambda x: np.sqrt((x["sq_diff"]).mean())
).reset_index(name="rmse")
print(f"   Node RMSE range: [{node_rmse['rmse'].min():.4f}, {node_rmse['rmse'].max():.4f}]")
print(f"   Worst nodes:")
print(node_rmse.nlargest(5, "rmse").to_string(index=False))

print("\n8. VARIANCE CHECK - Is Timer less variable?")
for mid in [1, 2]:
    mdf = merged[merged["model_id"] == mid]
    timer_var = mdf["timer_wl"].var()
    vgssm_var = mdf["vgssm_wl"].var()
    ratio = timer_var / vgssm_var
    print(f"   Model {mid}: Timer var={timer_var:.2f}, VGSSM var={vgssm_var:.2f}, ratio={ratio:.4f}")
