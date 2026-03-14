#!/usr/bin/env python3
"""
Validate submission format against sample_submission.

Checks:
1. All (model_id, event_id, node_type, node_id) combinations match
2. Row counts match
3. No missing values
4. Reasonable water level ranges
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys


def validate_submission(submission_path: str, sample_path: str = 'data/sample_submission.parquet'):
    """Validate submission against sample submission schema."""

    print("=" * 60)
    print("SUBMISSION VALIDATION")
    print("=" * 60)

    # Load files
    print(f"\nLoading sample submission: {sample_path}")
    sample = pd.read_parquet(sample_path)
    print(f"  Shape: {sample.shape}")

    print(f"\nLoading submission: {submission_path}")
    sub = pd.read_parquet(submission_path)
    print(f"  Shape: {sub.shape}")

    errors = []
    warnings = []

    # Check 1: Row count
    print("\n[1] Checking row count...")
    if len(sub) != len(sample):
        errors.append(f"Row count mismatch: submission has {len(sub):,}, expected {len(sample):,}")
    else:
        print(f"  OK: {len(sub):,} rows")

    # Check 2: Column names
    print("\n[2] Checking columns...")
    expected_cols = ['row_id', 'model_id', 'event_id', 'node_type', 'node_id', 'water_level']
    missing_cols = set(expected_cols) - set(sub.columns)
    extra_cols = set(sub.columns) - set(expected_cols)
    if missing_cols:
        errors.append(f"Missing columns: {missing_cols}")
    if extra_cols:
        warnings.append(f"Extra columns (will be ignored): {extra_cols}")
    if not missing_cols:
        print(f"  OK: All required columns present")

    # Check 3: row_id matches
    print("\n[3] Checking row_id alignment...")
    if 'row_id' in sub.columns and 'row_id' in sample.columns:
        if not (sub['row_id'] == sample['row_id']).all():
            # Check if they're the same set but different order
            if set(sub['row_id']) == set(sample['row_id']):
                warnings.append("row_id values match but ORDER differs - may cause scoring issues!")
            else:
                errors.append("row_id values don't match sample submission!")
        else:
            print("  OK: row_id values aligned")

    # Check 4: Key combinations match
    print("\n[4] Checking (model_id, event_id, node_type, node_id) combinations...")
    sample_keys = set(zip(sample['model_id'], sample['event_id'], sample['node_type'], sample['node_id']))
    sub_keys = set(zip(sub['model_id'], sub['event_id'], sub['node_type'], sub['node_id']))

    missing_keys = sample_keys - sub_keys
    extra_keys = sub_keys - sample_keys

    if missing_keys:
        errors.append(f"Missing {len(missing_keys):,} key combinations in submission")
        print(f"  Missing keys (first 5): {list(missing_keys)[:5]}")
    if extra_keys:
        errors.append(f"Extra {len(extra_keys):,} key combinations in submission")
        print(f"  Extra keys (first 5): {list(extra_keys)[:5]}")
    if not missing_keys and not extra_keys:
        print(f"  OK: All {len(sample_keys):,} key combinations match")

    # Check 5: Unique node_ids per model
    print("\n[5] Checking node_id ranges...")
    for model_id in [1, 2]:
        for node_type in [1, 2]:
            sample_nodes = sample[(sample['model_id'] == model_id) & (sample['node_type'] == node_type)]['node_id'].unique()
            sub_nodes = sub[(sub['model_id'] == model_id) & (sub['node_type'] == node_type)]['node_id'].unique()

            sample_min, sample_max = sample_nodes.min(), sample_nodes.max()
            sub_min, sub_max = sub_nodes.min() if len(sub_nodes) > 0 else -1, sub_nodes.max() if len(sub_nodes) > 0 else -1

            type_name = '1D' if node_type == 1 else '2D'
            print(f"  Model {model_id}, {type_name}: sample range [{sample_min}, {sample_max}] ({len(sample_nodes)} nodes)")
            print(f"                        sub range [{sub_min}, {sub_max}] ({len(sub_nodes)} nodes)")

            if set(sample_nodes) != set(sub_nodes):
                errors.append(f"Model {model_id} {type_name} node_ids don't match!")

    # Check 6: Missing values
    print("\n[6] Checking for missing values...")
    null_count = sub['water_level'].isna().sum()
    if null_count > 0:
        errors.append(f"{null_count:,} missing water_level values!")
    else:
        print("  OK: No missing values")

    # Check 7: Water level statistics
    print("\n[7] Checking water level ranges...")
    for model_id in [1, 2]:
        model_data = sub[sub['model_id'] == model_id]['water_level']
        print(f"  Model {model_id}: min={model_data.min():.2f}, max={model_data.max():.2f}, "
              f"mean={model_data.mean():.2f}, std={model_data.std():.2f}")

        if model_data.min() < 0:
            warnings.append(f"Model {model_id} has negative water levels (min={model_data.min():.2f})")
        if model_data.max() > 500:
            warnings.append(f"Model {model_id} has very high water levels (max={model_data.max():.2f})")

    # Check 8: Sample submission structure analysis
    print("\n[8] Sample submission structure analysis...")
    print(f"  Events per model:")
    for model_id in [1, 2]:
        events = sample[sample['model_id'] == model_id]['event_id'].unique()
        print(f"    Model {model_id}: {len(events)} events, IDs: {sorted(events)[:5]}...")

    print(f"\n  Timesteps per event (inferred from row count):")
    for model_id in [1, 2]:
        model_sample = sample[sample['model_id'] == model_id]
        events = model_sample['event_id'].unique()
        for event_id in sorted(events)[:2]:
            event_rows = model_sample[model_sample['event_id'] == event_id]
            n_1d = len(event_rows[event_rows['node_type'] == 1])
            n_2d = len(event_rows[event_rows['node_type'] == 2])
            n_1d_nodes = len(event_rows[event_rows['node_type'] == 1]['node_id'].unique())
            n_2d_nodes = len(event_rows[event_rows['node_type'] == 2]['node_id'].unique())
            timesteps_1d = n_1d // n_1d_nodes if n_1d_nodes > 0 else 0
            timesteps_2d = n_2d // n_2d_nodes if n_2d_nodes > 0 else 0
            print(f"    Model {model_id}, Event {event_id}: 1D={n_1d_nodes} nodes x {timesteps_1d} timesteps, "
                  f"2D={n_2d_nodes} nodes x {timesteps_2d} timesteps")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if errors:
        print(f"\n ERRORS ({len(errors)}):")
        for e in errors:
            print(f"  - {e}")

    if warnings:
        print(f"\n WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"  - {w}")

    if not errors and not warnings:
        print("\n ALL CHECKS PASSED!")

    print("=" * 60)

    return len(errors) == 0


if __name__ == '__main__':
    submission_path = sys.argv[1] if len(sys.argv) > 1 else 'submission_final.parquet'
    sample_path = sys.argv[2] if len(sys.argv) > 2 else 'data/sample_submission.parquet'

    success = validate_submission(submission_path, sample_path)
    sys.exit(0 if success else 1)
