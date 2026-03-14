#!/usr/bin/env python3
"""Analyze submission format and fix it to match expected format."""

import pandas as pd
import numpy as np

def analyze_formats():
    """Analyze sample submission vs our submission format."""
    sample = pd.read_parquet('data/sample_submission.parquet')
    our_sub = pd.read_parquet('submission_vgssm_physics_final.parquet')

    print('=== Sample Submission Analysis ===')
    print(f'Columns: {sample.columns.tolist()}')
    print(f'Shape: {sample.shape}')

    # Check structure for Model 1, Event 5
    m1_e5 = sample[(sample['model_id'] == 1) & (sample['event_id'] == 5)]
    m1_e5_n1 = m1_e5[m1_e5['node_type'] == 1]
    m1_e5_n2 = m1_e5[m1_e5['node_type'] == 2]

    n1_nodes = m1_e5_n1['node_id'].nunique()
    n2_nodes = m1_e5_n2['node_id'].nunique()
    n1_rows_per_node = len(m1_e5_n1) // max(1, n1_nodes)
    n2_rows_per_node = len(m1_e5_n2) // max(1, n2_nodes)

    print(f'\nModel 1, Event 5:')
    print(f'  Node type 1 (1D): {n1_nodes} nodes x {n1_rows_per_node} timesteps = {len(m1_e5_n1)} rows')
    print(f'  Node type 2 (2D): {n2_nodes} nodes x {n2_rows_per_node} timesteps = {len(m1_e5_n2)} rows')

    print('\n=== Our Submission Analysis ===')
    print(f'Columns: {our_sub.columns.tolist()}')
    print(f'Shape: {our_sub.shape}')

    o1_e5 = our_sub[(our_sub['model_id'] == 1) & (our_sub['event_id'] == 5)]
    o1_e5_n1 = o1_e5[o1_e5['node_type'] == '1d']
    o1_e5_n2 = o1_e5[o1_e5['node_type'] == '2d']

    on1_nodes = o1_e5_n1['node_idx'].nunique()
    on2_nodes = o1_e5_n2['node_idx'].nunique()

    print(f'\nModel 1, Event 5:')
    print(f'  Node type 1d: {on1_nodes} nodes, timesteps {o1_e5_n1["timestep"].min()}-{o1_e5_n1["timestep"].max()}')
    print(f'  Node type 2d: {on2_nodes} nodes, timesteps {o1_e5_n2["timestep"].min()}-{o1_e5_n2["timestep"].max()}')
    print(f'  Total rows: {len(o1_e5)}')

    print('\n=== Format Differences ===')
    print(f'Sample expected: {len(sample):,} rows')
    print(f'Our submission: {len(our_sub):,} rows')
    print(f'Missing rows: {len(sample) - len(our_sub):,}')

    # Check event coverage
    print('\n=== Event Coverage ===')
    sample_events_m1 = set(sample[sample['model_id'] == 1]['event_id'].unique())
    sample_events_m2 = set(sample[sample['model_id'] == 2]['event_id'].unique())
    our_events_m1 = set(our_sub[our_sub['model_id'] == 1]['event_id'].unique())
    our_events_m2 = set(our_sub[our_sub['model_id'] == 2]['event_id'].unique())

    print(f'Sample Model 1 events: {sorted(sample_events_m1)}')
    print(f'Our Model 1 events: {sorted(our_events_m1)}')
    print(f'Missing Model 1 events: {sorted(sample_events_m1 - our_events_m1)}')
    print()
    print(f'Sample Model 2 events: {sorted(sample_events_m2)}')
    print(f'Our Model 2 events: {sorted(our_events_m2)}')
    print(f'Missing Model 2 events: {sorted(sample_events_m2 - our_events_m2)}')

    return sample, our_sub


def fix_submission_format(our_sub, sample):
    """Fix our submission to match the expected format."""
    print('\n=== Fixing Submission Format ===')

    # Create a mapping from sample submission to get row_id assignments
    # The sample submission has row_id ordered by: model_id, event_id, node_type, node_id, timestep (implicit)

    # First, let's understand the exact ordering in sample
    sample_sorted = sample.sort_values('row_id')

    # Check if ordering is consistent
    first_100 = sample_sorted.head(200)[['row_id', 'model_id', 'event_id', 'node_type', 'node_id']]
    print('Sample submission ordering (first 200 rows):')
    print(first_100)

    return None


if __name__ == '__main__':
    sample, our_sub = analyze_formats()
    fix_submission_format(our_sub, sample)
