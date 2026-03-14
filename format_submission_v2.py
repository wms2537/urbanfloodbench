#!/usr/bin/env python3
"""Format submission to match expected format - memory efficient version."""

import pandas as pd
import numpy as np
import gc

print("Loading sample submission...")
sample = pd.read_parquet('data/sample_submission.parquet')
print(f"Sample shape: {sample.shape}")

# Create merge key for sample
print("Creating sample merge keys...")
sample['merge_key'] = (
    sample['model_id'].astype(str) + '_' +
    sample['event_id'].astype(str) + '_' +
    sample['node_type'].astype(str) + '_' +
    sample['node_id'].astype(str)
)
sample['timestep_idx'] = sample.groupby('merge_key').cumcount()

print("Processing Model 1...")
pred1 = pd.read_parquet('submission_graph_tft_model1.parquet')
print(f"M1 shape: {pred1.shape}")

# Map element_type to node_type
pred1['node_type'] = pred1['element_type'].map({'1d': 1, '2d': 2})
pred1['node_id'] = pred1['element_id']

# Remove warmup timesteps
warmup = 10
pred1 = pred1[pred1['timestep'] >= warmup].copy()
pred1['timestep_adj'] = pred1['timestep'] - warmup

# Create merge key
pred1['merge_key'] = (
    pred1['model_id'].astype(int).astype(str) + '_' +
    pred1['event_id'].astype(int).astype(str) + '_' +
    pred1['node_type'].astype(int).astype(str) + '_' +
    pred1['node_id'].astype(int).astype(str)
)

# Keep only needed columns
pred1 = pred1[['merge_key', 'timestep_adj', 'water_level']]
gc.collect()

print("Processing Model 2...")
pred2 = pd.read_parquet('submission_graph_tft_model2.parquet')
print(f"M2 shape: {pred2.shape}")

pred2['node_type'] = pred2['element_type'].map({'1d': 1, '2d': 2})
pred2['node_id'] = pred2['element_id']
pred2 = pred2[pred2['timestep'] >= warmup].copy()
pred2['timestep_adj'] = pred2['timestep'] - warmup
pred2['merge_key'] = (
    pred2['model_id'].astype(int).astype(str) + '_' +
    pred2['event_id'].astype(int).astype(str) + '_' +
    pred2['node_type'].astype(int).astype(str) + '_' +
    pred2['node_id'].astype(int).astype(str)
)
pred2 = pred2[['merge_key', 'timestep_adj', 'water_level']]
gc.collect()

print("Concatenating predictions...")
pred = pd.concat([pred1, pred2], ignore_index=True)
del pred1, pred2
gc.collect()
print(f"Combined pred shape: {pred.shape}")

print("Merging...")
merged = sample.merge(
    pred,
    left_on=['merge_key', 'timestep_idx'],
    right_on=['merge_key', 'timestep_adj'],
    how='left',
    suffixes=('_sample', '')
)
del pred
gc.collect()

print(f"Merged shape: {merged.shape}")
print(f"Missing water_level: {merged['water_level'].isna().sum():,}")

# Create final submission
final = merged[['row_id', 'model_id', 'event_id', 'node_type', 'node_id', 'water_level']].copy()
del merged
gc.collect()

final['water_level'] = final['water_level'].fillna(0)

# Save
print("Saving...")
final.to_parquet('submission_final.parquet', index=False)
print(f"\nFinal submission shape: {final.shape}")
print(f"File saved to submission_final.parquet")
print(final['water_level'].describe())
