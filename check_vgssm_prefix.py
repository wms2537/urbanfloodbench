#!/usr/bin/env python3
"""Check if VGSSM has prefix data."""

import pandas as pd

vgssm = pd.read_parquet("submission_vgssm_v1_final.parquet")
sample = pd.read_parquet("data/sample_submission.parquet")
sample = sample.sort_values("row_id").reset_index(drop=True)
sample["timestep"] = sample.groupby(["model_id", "event_id", "node_type", "node_id"]).cumcount()

vgssm = vgssm.merge(sample[["row_id", "timestep"]], on="row_id")

# Check prefix (t<10) values for Model 2
prefix = vgssm[(vgssm["model_id"] == 2) & (vgssm["timestep"] < 10)]
wl = prefix["water_level"]
print(f"Model 2 prefix (t<10): water_level [{wl.min():.2f}, {wl.max():.2f}]")
print("(Ground truth range is approximately 23-47)")
