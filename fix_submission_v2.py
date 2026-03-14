#!/usr/bin/env python3
"""Fix submission format - v2 with proper row alignment."""

import pandas as pd
import numpy as np
from pathlib import Path


def fix_submission(our_submission_path, sample_submission_path, output_path):
    """Fix our submission to match the expected format - correct version."""
    print("Loading sample submission to get structure...")
    sample = pd.read_parquet(sample_submission_path)
    print(f"Sample submission: {len(sample):,} rows")

    print("Loading our submission...")
    our_sub = pd.read_parquet(our_submission_path)
    print(f"Our submission: {len(our_sub):,} rows")

    # Check columns
    print(f"\nOur columns: {our_sub.columns.tolist()}")
    print(f"Sample columns: {sample.columns.tolist()}")

    # Fix column names
    if 'node_idx' in our_sub.columns:
        our_sub = our_sub.rename(columns={'node_idx': 'node_id'})

    # Fix node_type values (string to int)
    if our_sub['node_type'].dtype == object:
        our_sub['node_type'] = our_sub['node_type'].map({'1d': 1, '2d': 2})

    print(f"\nOur timestep range: {our_sub['timestep'].min()} - {our_sub['timestep'].max()}")
    print(f"Our water_level range: {our_sub['water_level'].min():.4f} - {our_sub['water_level'].max():.4f}")

    # Check data by model
    for model_id in our_sub['model_id'].unique():
        model_data = our_sub[our_sub['model_id'] == model_id]
        print(f"\nModel {model_id}:")
        print(f"  Rows: {len(model_data):,}")
        print(f"  water_level range: {model_data['water_level'].min():.4f} - {model_data['water_level'].max():.4f}")
        print(f"  node_type values: {model_data['node_type'].unique()}")

    # The sample submission has a specific row order
    # Each row in sample corresponds to a specific (model, event, node_type, node, timestep)
    # We need to figure out the timestep for each row in sample

    # Add timestep to sample by computing it within each group
    print("\nAdding timestep to sample structure...")
    sample = sample.sort_values('row_id').reset_index(drop=True)

    # Compute timestep as position within each (model, event, node_type, node) group
    sample['timestep'] = sample.groupby(['model_id', 'event_id', 'node_type', 'node_id']).cumcount()

    print(f"Sample timestep range: {sample['timestep'].min()} - {sample['timestep'].max()}")

    # Now merge our predictions with sample based on all keys including timestep
    print("\nMerging predictions...")

    # Keep only the columns we need from sample
    sample_keys = sample[['row_id', 'model_id', 'event_id', 'node_type', 'node_id', 'timestep']].copy()

    # Merge
    merged = sample_keys.merge(
        our_sub[['model_id', 'event_id', 'node_type', 'node_id', 'timestep', 'water_level']],
        on=['model_id', 'event_id', 'node_type', 'node_id', 'timestep'],
        how='left'
    )

    print(f"Merged rows: {len(merged):,}")
    print(f"Missing predictions: {merged['water_level'].isna().sum():,}")

    # Check missing by model
    for model_id in [1, 2]:
        model_data = merged[merged['model_id'] == model_id]
        missing = model_data['water_level'].isna().sum()
        print(f"  Model {model_id}: {missing:,} missing out of {len(model_data):,}")

    # Fill missing with last known value per group or 0
    if merged['water_level'].isna().any():
        print("\nFilling missing values...")
        merged = merged.sort_values(['model_id', 'event_id', 'node_type', 'node_id', 'timestep'])
        merged['water_level'] = merged.groupby(['model_id', 'event_id', 'node_type', 'node_id'])['water_level'].ffill()
        merged['water_level'] = merged['water_level'].fillna(0.0)

    # Sort back by row_id
    merged = merged.sort_values('row_id').reset_index(drop=True)

    # Create final submission
    final_df = merged[['row_id', 'model_id', 'event_id', 'node_type', 'node_id', 'water_level']].copy()

    # Validate
    print("\n=== Validation ===")
    print(f"Final shape: {final_df.shape}")
    print(f"Columns: {final_df.columns.tolist()}")

    for model_id in [1, 2]:
        model_data = final_df[final_df['model_id'] == model_id]
        print(f"\nModel {model_id}:")
        print(f"  Rows: {len(model_data):,}")
        print(f"  water_level: min={model_data['water_level'].min():.4f}, max={model_data['water_level'].max():.4f}, mean={model_data['water_level'].mean():.4f}")

    # Compare with sample structure
    print("\n=== Structure Check ===")
    print(f"row_id matches: {(final_df['row_id'] == sample['row_id']).all()}")
    print(f"model_id matches: {(final_df['model_id'] == sample['model_id']).all()}")
    print(f"event_id matches: {(final_df['event_id'] == sample['event_id']).all()}")
    print(f"node_type matches: {(final_df['node_type'] == sample['node_type']).all()}")
    print(f"node_id matches: {(final_df['node_id'] == sample['node_id']).all()}")

    # Save
    final_df.to_parquet(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    return final_df


if __name__ == '__main__':
    fix_submission(
        our_submission_path='submission_vgssm_physics_final.parquet',
        sample_submission_path='data/sample_submission.parquet',
        output_path='submission_fixed_v2.parquet'
    )
