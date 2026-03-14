#!/usr/bin/env python3
"""
Full pipeline: Train both models → Generate predictions → Format submission
All in one script, runs sequentially.

Usage:
    python run_full_pipeline.py --exp_name graph_tft_v5 --hidden_dim 128 --loss_type huber --patience 20
"""

import argparse
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and wait for completion."""
    print(f"\n{'='*60}")
    print(f"[RUNNING] {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=Path(__file__).parent)

    if result.returncode != 0:
        print(f"\n[ERROR] {description} failed with code {result.returncode}")
        sys.exit(1)

    print(f"\n[DONE] {description}")
    return result

def main():
    parser = argparse.ArgumentParser(description='Full training and prediction pipeline')

    # Experiment
    parser.add_argument('--exp_name', type=str, required=True, help='Experiment name')

    # Model architecture
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--loss_type', type=str, default='mse', choices=['mse', 'huber', 'weighted_mse'])
    parser.add_argument('--patience', type=int, default=7, help='Early stopping patience')

    # Training
    parser.add_argument('--max_epochs', type=int, default=60, help='Max epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for Model 1')
    parser.add_argument('--batch_size_m2', type=int, default=None, help='Batch size for Model 2 (default: same as batch_size)')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1, help='Gradient accumulation for Model 1')
    parser.add_argument('--accumulate_grad_batches_m2', type=int, default=None, help='Gradient accumulation for Model 2')

    # Prediction
    parser.add_argument('--calibration_steps', type=int, default=100, help='Calibration steps')

    # Skip options
    parser.add_argument('--skip_m1_train', action='store_true', help='Skip Model 1 training')
    parser.add_argument('--skip_m2_train', action='store_true', help='Skip Model 2 training')
    parser.add_argument('--skip_predict', action='store_true', help='Skip prediction')

    args = parser.parse_args()

    # Default M2 settings
    if args.batch_size_m2 is None:
        args.batch_size_m2 = args.batch_size
    if args.accumulate_grad_batches_m2 is None:
        args.accumulate_grad_batches_m2 = args.accumulate_grad_batches

    print("="*60)
    print("FULL PIPELINE")
    print("="*60)
    print(f"Experiment: {args.exp_name}")
    print(f"hidden_dim: {args.hidden_dim}")
    print(f"loss_type: {args.loss_type}")
    print(f"patience: {args.patience}")
    print(f"max_epochs: {args.max_epochs}")
    print(f"Model 1: batch={args.batch_size}, accum={args.accumulate_grad_batches}")
    print(f"Model 2: batch={args.batch_size_m2}, accum={args.accumulate_grad_batches_m2}")
    print(f"calibration_steps: {args.calibration_steps}")
    print("="*60)

    # Common training args
    common_args = [
        '--exp_name', args.exp_name,
        '--max_epochs', str(args.max_epochs),
        '--hidden_dim', str(args.hidden_dim),
        '--loss_type', args.loss_type,
        '--patience', str(args.patience),
        '--accelerator', 'cuda',
        '--data_dir', './data',
    ]

    # Train Model 1
    if not args.skip_m1_train:
        cmd = [
            sys.executable, 'train_graph_tft.py',
            '--model_id', '1',
            '--batch_size', str(args.batch_size),
            '--accumulate_grad_batches', str(args.accumulate_grad_batches),
        ] + common_args
        run_command(cmd, "Training Model 1")

    # Train Model 2
    if not args.skip_m2_train:
        cmd = [
            sys.executable, 'train_graph_tft.py',
            '--model_id', '2',
            '--batch_size', str(args.batch_size_m2),
            '--accumulate_grad_batches', str(args.accumulate_grad_batches_m2),
        ] + common_args
        run_command(cmd, "Training Model 2")

    if args.skip_predict:
        print("\n[SKIPPED] Prediction")
        return

    # Find best checkpoints
    print("\n[INFO] Finding best checkpoints...")
    ckpt_dir = Path('checkpoints')
    import re

    def find_best_ckpt(model_id):
        """Find checkpoint with lowest validation metric (sorted by metric in filename)."""
        model_dir = ckpt_dir / f'model_{model_id}' / args.exp_name
        ckpts = [c for c in model_dir.glob('*.ckpt') if c.name != 'last.ckpt']

        if not ckpts:
            return str(model_dir / 'last.ckpt')

        def extract_metric(path):
            # Filename format: epoch=XX-val/std_rmse=Y.YYYY.ckpt
            # or: XX-Y.YYYY.ckpt
            m = re.search(r'[=-]([0-9]+\.[0-9]+)\.ckpt$', path.name)
            if m:
                return float(m.group(1))
            return float('inf')

        # Sort by metric (ascending - lower is better)
        ckpts = sorted(ckpts, key=extract_metric)
        print(f"  Model {model_id}: Found {len(ckpts)} checkpoints, best metric: {extract_metric(ckpts[0]):.6f}")
        return str(ckpts[0])

    m1_ckpt = find_best_ckpt(1)
    m2_ckpt = find_best_ckpt(2)
    print(f"Model 1: {m1_ckpt}")
    print(f"Model 2: {m2_ckpt}")

    # Compute ACTUAL hidden_dim used during training
    # Training script uses: max(args.hidden_dim, 96) for Model 2
    m1_hidden_dim = args.hidden_dim
    m2_hidden_dim = max(args.hidden_dim, 96)  # Match training logic!

    print(f"Model 1 hidden_dim (for prediction): {m1_hidden_dim}")
    print(f"Model 2 hidden_dim (for prediction): {m2_hidden_dim}")

    # Predict Model 1
    cmd = [
        sys.executable, 'predict_graph_tft.py',
        '--model_id', '1',
        '--checkpoint', m1_ckpt,
        '--hidden_dim', str(m1_hidden_dim),
        '--calibrate_latent',
        '--calibration_steps', str(args.calibration_steps),
    ]
    run_command(cmd, "Predicting Model 1")

    # Predict Model 2 - use ACTUAL trained hidden_dim
    cmd = [
        sys.executable, 'predict_graph_tft.py',
        '--model_id', '2',
        '--checkpoint', m2_ckpt,
        '--hidden_dim', str(m2_hidden_dim),  # CRITICAL: must match training!
        '--calibrate_latent',
        '--calibration_steps', str(args.calibration_steps),
    ]
    run_command(cmd, "Predicting Model 2")

    # Format submission
    cmd = [sys.executable, 'format_submission_v2.py']
    run_command(cmd, "Formatting submission")

    # Print final stats
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print("="*60)

    import pandas as pd
    final = pd.read_parquet('submission_final.parquet')
    print(f"Submission: {final.shape[0]:,} rows")
    print(f"File size: {Path('submission_final.parquet').stat().st_size / 1e6:.1f} MB")
    print(final['water_level'].describe())
    print("="*60)

if __name__ == '__main__':
    main()
