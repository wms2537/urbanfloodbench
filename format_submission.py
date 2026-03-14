#!/usr/bin/env python3
"""Format submission to match expected format."""

import pandas as pd
import numpy as np
import sys

# Load sample submission template
sample = pd.read_parquet('data/sample_submission.parquet')

# Load our predictions - try v3 files first, fall back to v2
try:
    pred1 = pd.read_parquet('submission_graph_tft_model1.parquet')
    pred2 = pd.read_parquet('submission_graph_tft_model2.parquet')
    pred = pd.concat([pred1, pred2], ignore_index=True)
    print(f'Loaded v3 predictions: M1={len(pred1):,}, M2={len(pred2):,}')
except FileNotFoundError:
    pred = pd.read_parquet('submission_graph_tft_v2_calibrated.parquet')
    print(f'Loaded v2 predictions: {len(pred):,}')

# Map element_type to node_type
pred['node_type'] = pred['element_type'].map({'1d': 1, '2d': 2})
pred['node_id'] = pred['element_id']

# Remove warmup timesteps (first 10)
warmup = 10
pred_filtered = pred[pred['timestep'] >= warmup].copy()

# Adjust timestep to match sample (0-indexed from prediction start)
pred_filtered['timestep_adj'] = pred_filtered['timestep'] - warmup

print(f'Original rows: {len(pred):,}')
print(f'After removing warmup: {len(pred_filtered):,}')
print(f'Sample rows: {len(sample):,}')

# Create merge key for sample
sample['merge_key'] = (
    sample['model_id'].astype(str) + '_' +
    sample['event_id'].astype(str) + '_' +
    sample['node_type'].astype(str) + '_' +
    sample['node_id'].astype(str)
)

# Add row number within each group as timestep
sample['timestep_idx'] = sample.groupby('merge_key').cumcount()

print()
print('Sample timestep counts:')
print(sample.groupby('merge_key')['timestep_idx'].max().value_counts().head())

# Create merge key for predictions
pred_filtered['merge_key'] = (
    pred_filtered['model_id'].astype(int).astype(str) + '_' +
    pred_filtered['event_id'].astype(int).astype(str) + '_' +
    pred_filtered['node_type'].astype(int).astype(str) + '_' +
    pred_filtered['node_id'].astype(int).astype(str)
)

print()
print('Prediction timestep counts:')
print(pred_filtered.groupby('merge_key')['timestep_adj'].max().value_counts().head())

# Merge
merged = sample.merge(
    pred_filtered[['merge_key', 'timestep_adj', 'water_level']],
    left_on=['merge_key', 'timestep_idx'],
    right_on=['merge_key', 'timestep_adj'],
    how='left',
    suffixes=('_sample', '')
)

print()
print(f'Merged rows: {len(merged):,}')
print(f'Missing water_level: {merged["water_level"].isna().sum():,}')

# Create final submission
final = merged[['row_id', 'model_id', 'event_id', 'node_type', 'node_id', 'water_level']].copy()
final['water_level'] = final['water_level'].fillna(0)  # Fill any missing with 0

# Save
final.to_parquet('submission_final.parquet', index=False)
print()
print(f'Final submission shape: {final.shape}')
print(f'Final submission dtypes:')
print(final.dtypes)
print()
print('Stats:')
print(final['water_level'].describe())
print()
print('Stats by model:')
print(final.groupby('model_id')['water_level'].describe())
