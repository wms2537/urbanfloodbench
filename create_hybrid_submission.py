#!/usr/bin/env python3
"""Create hybrid submission: Timer V4 for Model 1, baseline VGSSM for Model 2."""

import pandas as pd
import numpy as np

print("=== Creating Hybrid Submission ===\n")

# Load Timer V4 submission
timer = pd.read_parquet("submission_final_timer_v4.parquet")
print(f"Timer V4: {len(timer):,} rows")

# Load baseline VGSSM submission (the one that scored 0.18)
vgssm = pd.read_parquet("submission_vgssm_v1_final.parquet")
print(f"Baseline VGSSM: {len(vgssm):,} rows")

# Create hybrid: Model 1 from Timer, Model 2 from VGSSM
timer_m1 = timer[timer["model_id"] == 1].copy()
vgssm_m2 = vgssm[vgssm["model_id"] == 2].copy()

print(f"\nHybrid composition:")
print(f"  Model 1 (Timer V4): {len(timer_m1):,} rows")
print(f"  Model 2 (VGSSM):    {len(vgssm_m2):,} rows")

# Combine
hybrid = pd.concat([timer_m1, vgssm_m2], ignore_index=True)
hybrid = hybrid.sort_values("row_id").reset_index(drop=True)

print(f"\nHybrid total: {len(hybrid):,} rows")

# Validate
print("\n=== Validation ===")
for mid in [1, 2]:
    mdf = hybrid[hybrid["model_id"] == mid]
    wl = mdf["water_level"]
    print(f"Model {mid}: {len(mdf):,} rows, water [{wl.min():.2f}, {wl.max():.2f}], mean={wl.mean():.2f}")

# Check structure matches
sample = pd.read_parquet("data/sample_submission.parquet")
print(f"\nStructure check:")
print(f"  row_id matches: {(hybrid['row_id'] == sample['row_id']).all()}")
print(f"  model_id matches: {(hybrid['model_id'] == sample['model_id']).all()}")

# Save
hybrid.to_parquet("submission_hybrid_v1.parquet", index=False)
print(f"\nSaved to: submission_hybrid_v1.parquet")
print(f"Size: {hybrid.memory_usage(deep=True).sum() / 1e6:.1f} MB")
