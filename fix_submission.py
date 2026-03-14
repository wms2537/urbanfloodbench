#!/usr/bin/env python3
"""Fix submission format to match expected competition format - optimized version."""

import pandas as pd
import numpy as np
from pathlib import Path


def fix_submission(our_submission_path, sample_submission_path, output_path):
    """Fix our submission to match the expected format - memory efficient version."""
    print("Loading sample submission to get structure...")
    sample = pd.read_parquet(sample_submission_path)
    print(f"Sample submission: {len(sample):,} rows")

    # Keep only the structure columns, drop water_level to save memory
    sample_structure = sample[['row_id', 'model_id', 'event_id', 'node_type', 'node_id']].copy()
    del sample

    print("Loading our submission...")
    our_sub = pd.read_parquet(our_submission_path)
    print(f"Our submission: {len(our_sub):,} rows")

    # Fix column names and values
    our_sub = our_sub.rename(columns={'node_idx': 'node_id'})
    our_sub['node_type'] = our_sub['node_type'].map({'1d': 1, '2d': 2})

    print(f"Our timestep range: {our_sub['timestep'].min()} - {our_sub['timestep'].max()}")

    # Create a mapping from (model_id, event_id, node_type, node_id, relative_timestep) -> water_level
    # The sample submission orders by row_id, which implicitly orders by timestep
    print("\nCreating prediction lookup...")

    # For each (model_id, event_id, node_type, node_id), our predictions are sorted by timestep
    # We need to map these to the sample submission's row ordering

    # First, get counts for each group in sample
    group_counts = sample_structure.groupby(['model_id', 'event_id', 'node_type', 'node_id']).size().reset_index(name='n_timesteps')
    print(f"Number of unique (model, event, node_type, node) groups: {len(group_counts):,}")

    # Sort our predictions properly
    our_sub = our_sub.sort_values(['model_id', 'event_id', 'node_type', 'node_id', 'timestep'])

    # Build water_level array aligned to sample structure
    print("\nBuilding water level array...")

    # Merge to get expected n_timesteps for each of our predictions
    our_sub = our_sub.merge(
        group_counts,
        on=['model_id', 'event_id', 'node_type', 'node_id'],
        how='left'
    )

    # For each group, we need to pad/truncate to n_timesteps
    results = []
    for (model_id, event_id, node_type, node_id), group in our_sub.groupby(
        ['model_id', 'event_id', 'node_type', 'node_id'], sort=True
    ):
        n_expected = int(group['n_timesteps'].iloc[0])
        water_levels = group['water_level'].values

        if len(water_levels) < n_expected:
            # Pad with last value
            last_val = water_levels[-1] if len(water_levels) > 0 else 0.0
            padded = np.concatenate([water_levels, np.full(n_expected - len(water_levels), last_val)])
        elif len(water_levels) > n_expected:
            # Truncate
            padded = water_levels[:n_expected]
        else:
            padded = water_levels

        results.append({
            'model_id': model_id,
            'event_id': event_id,
            'node_type': node_type,
            'node_id': node_id,
            'water_levels': padded
        })

    print(f"Processed {len(results):,} groups")

    # Now expand to match sample structure exactly
    print("\nExpanding to final format...")

    # Sort sample structure
    sample_structure = sample_structure.sort_values('row_id')

    # Create a lookup dict for water levels
    water_level_dict = {}
    for r in results:
        key = (r['model_id'], r['event_id'], r['node_type'], r['node_id'])
        water_level_dict[key] = r['water_levels']

    # Process sample structure group by group
    final_water_levels = []
    missing_groups = []

    for (model_id, event_id, node_type, node_id), group in sample_structure.groupby(
        ['model_id', 'event_id', 'node_type', 'node_id'], sort=False
    ):
        key = (model_id, event_id, node_type, node_id)
        n_rows = len(group)

        if key in water_level_dict:
            wl = water_level_dict[key]
            if len(wl) != n_rows:
                print(f"WARNING: Size mismatch for {key}: expected {n_rows}, got {len(wl)}")
                if len(wl) < n_rows:
                    wl = np.concatenate([wl, np.full(n_rows - len(wl), wl[-1] if len(wl) > 0 else 0.0)])
                else:
                    wl = wl[:n_rows]
        else:
            missing_groups.append(key)
            wl = np.zeros(n_rows)

        final_water_levels.extend(wl.tolist())

    if missing_groups:
        print(f"WARNING: {len(missing_groups)} missing groups")
        print(f"First 5: {missing_groups[:5]}")

    # Create final dataframe
    print("\nCreating final submission...")
    sample_structure['water_level'] = final_water_levels

    # Ensure correct column order
    final_df = sample_structure[['row_id', 'model_id', 'event_id', 'node_type', 'node_id', 'water_level']]

    print(f"Final submission: {len(final_df):,} rows")

    # Save
    final_df.to_parquet(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    # Quick validation
    print("\nValidation:")
    print(f"  Columns: {final_df.columns.tolist()}")
    print(f"  Shape: {final_df.shape}")
    print(f"  Sample head:")
    print(final_df.head(10))
    print(f"  Water level stats: min={final_df['water_level'].min():.4f}, max={final_df['water_level'].max():.4f}, mean={final_df['water_level'].mean():.4f}")

    return final_df


if __name__ == '__main__':
    fix_submission(
        our_submission_path='submission_vgssm_physics_final.parquet',
        sample_submission_path='data/sample_submission.parquet',
        output_path='submission_fixed.parquet'
    )
