#!/usr/bin/env python3
"""
Prediction script for VGSSM with test-time latent calibration.

Uses warmup period to calibrate both event latent (c_e) and initial state (z_0)
before rolling out predictions for the full horizon.

Usage:
    python predict_vgssm.py \
        --model_id 1 \
        --checkpoint checkpoints/model_1/vgssm_v1/best.ckpt \
        --calibrate_latent \
        --calibration_steps 50
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.dataset import FloodEventDataset
from src.data.graph_builder import FloodGraphBuilder
from src.models.vgssm import VGSSM
from src.training.vgssm_trainer import VGSSMTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Generate predictions with VGSSM')

    # Data
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to data directory')
    parser.add_argument('--model_id', type=int, required=True, choices=[1, 2],
                        help='Model ID (1 or 2)')

    # Model
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')

    # Prediction settings
    parser.add_argument('--prefix_len', type=int, default=10,
                        help='Prefix length for encoding')
    parser.add_argument('--warmup_len', type=int, default=5,
                        help='Warmup steps for calibration (uses known observations)')

    # Test-time calibration
    parser.add_argument('--calibrate_latent', action='store_true',
                        help='Enable test-time latent calibration')
    parser.add_argument('--calibration_steps', type=int, default=50,
                        help='Number of calibration optimization steps')
    parser.add_argument('--calibration_lr', type=float, default=0.01,
                        help='Learning rate for calibration')

    # Output
    parser.add_argument('--output_dir', type=str, default='predictions',
                        help='Output directory for predictions')
    parser.add_argument('--output_suffix', type=str, default='',
                        help='Suffix for output filename')

    # Hardware
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cuda, mps, cpu)')

    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    """Get torch device from string."""
    if device_str == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    return torch.device(device_str)


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[VGSSM, Dict]:
    """Load VGSSM model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract hyperparameters
    hparams = checkpoint.get('hyper_parameters', {})

    # Get model state dict
    state_dict = checkpoint['state_dict']

    # Remove 'model.' prefix if present
    model_state = {}
    for k, v in state_dict.items():
        if k.startswith('model.'):
            model_state[k[6:]] = v
        else:
            model_state[k] = v

    return model_state, hparams


def calibrate_latents(
    model: VGSSM,
    graph,
    prefix_1d: torch.Tensor,
    prefix_2d: torch.Tensor,
    warmup_target_1d: torch.Tensor,
    warmup_target_2d: torch.Tensor,
    warmup_rainfall: torch.Tensor,
    num_steps: int = 50,
    lr: float = 0.01,
    device: torch.device = torch.device('cpu'),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Test-time calibration of latent states.

    Optimizes c_e, z0_1d, z0_2d to minimize prediction error on warmup period.

    Args:
        model: VGSSM model
        graph: HeteroData graph
        prefix_1d: [1, prefix_len, num_1d, dynamic_1d_dim]
        prefix_2d: [1, prefix_len, num_2d, dynamic_2d_dim]
        warmup_target_1d: [1, warmup_len, num_1d] water levels
        warmup_target_2d: [1, warmup_len, num_2d] water levels
        warmup_rainfall: [1, warmup_len, num_2d, 1]
        num_steps: Optimization steps
        lr: Learning rate

    Returns:
        Optimized (c_e, z0_1d, z0_2d)
    """
    model.eval()

    # Initialize from posterior
    with torch.no_grad():
        spatial_1d, spatial_2d = model.encode_spatial(graph)
        c_e_init, _, _ = model.encode_event_latent(prefix_1d, prefix_2d)
        z0_mu_1d, _ = model.z0_encoder_1d(prefix_1d, spatial_1d, c_e_init)
        z0_mu_2d, _ = model.z0_encoder_2d(prefix_2d, spatial_2d, c_e_init)

    # Make optimizable
    c_e = c_e_init.clone().requires_grad_(True)
    z0_1d = z0_mu_1d.clone().requires_grad_(True)
    z0_2d = z0_mu_2d.clone().requires_grad_(True)

    optimizer = torch.optim.Adam([c_e, z0_1d, z0_2d], lr=lr)

    warmup_len = warmup_target_1d.shape[1]

    # Enable gradients for optimization
    model.train()

    for step in range(num_steps):
        optimizer.zero_grad()

        # Forward with current latents
        outputs = model.forward_from_latents(
            graph, c_e, z0_1d, z0_2d,
            warmup_rainfall,
        )

        # Loss on warmup period
        pred_1d = outputs['pred_1d'][:, :warmup_len, :, 0]
        pred_2d = outputs['pred_2d'][:, :warmup_len, :, 0]

        loss_1d = F.mse_loss(pred_1d, warmup_target_1d)
        loss_2d = F.mse_loss(pred_2d, warmup_target_2d)

        # Regularization toward prior
        kl_ce = 0.01 * (c_e ** 2).mean()
        kl_z0 = 0.001 * (z0_1d ** 2).mean() + 0.001 * (z0_2d ** 2).mean()

        loss = loss_1d + loss_2d + kl_ce + kl_z0
        loss.backward()
        optimizer.step()

    model.eval()
    return c_e.detach(), z0_1d.detach(), z0_2d.detach()


def predict_event(
    model: VGSSM,
    graph,
    dataset: FloodEventDataset,
    prefix_len: int,
    warmup_len: int,
    calibrate: bool,
    calibration_steps: int,
    calibration_lr: float,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Generate predictions for a single event.

    Args:
        model: VGSSM model
        graph: HeteroData graph
        dataset: FloodEventDataset for this event
        prefix_len: Prefix length for encoding
        warmup_len: Warmup length for calibration
        calibrate: Whether to do test-time calibration
        calibration_steps: Number of calibration steps
        calibration_lr: Calibration learning rate
        device: Torch device

    Returns:
        Dict with predictions
    """
    model.eval()

    # Get full event sequence
    full_seq = dataset.get_full_sequence()
    input_1d = full_seq['input_1d'].unsqueeze(0).to(device)  # [1, T, N_1d, D]
    input_2d = full_seq['input_2d'].unsqueeze(0).to(device)  # [1, T, N_2d, D]

    num_timesteps = full_seq['num_timesteps']
    num_1d_nodes = input_1d.shape[2]
    num_2d_nodes = input_2d.shape[2]

    # Get raw water levels for calibration targets
    water_level_1d = full_seq['water_level_1d'].numpy()  # [T, N_1d, 1]
    water_level_2d = full_seq['water_level_2d'].numpy()  # [T, N_2d, 1]

    # Normalize targets if dataset has normalization
    if dataset.norm_stats is not None:
        wl_1d_norm = (water_level_1d - dataset.norm_stats['target_1d']['mean']) / dataset.norm_stats['target_1d']['std']
        wl_2d_norm = (water_level_2d - dataset.norm_stats['target_2d']['mean']) / dataset.norm_stats['target_2d']['std']
    else:
        wl_1d_norm = water_level_1d
        wl_2d_norm = water_level_2d

    # Predictions storage
    preds_1d = np.zeros((num_timesteps, num_1d_nodes))
    preds_2d = np.zeros((num_timesteps, num_2d_nodes))

    # Start from prefix_len
    start_idx = prefix_len + warmup_len
    horizon = num_timesteps - start_idx

    if horizon <= 0:
        print(f"  Warning: Event too short for prediction (timesteps={num_timesteps})")
        return {
            'pred_1d': preds_1d,
            'pred_2d': preds_2d,
            'event_id': dataset.event_id,
        }

    # Extract sequences
    prefix_1d = input_1d[:, :prefix_len]
    prefix_2d = input_2d[:, :prefix_len]

    # Warmup targets and rainfall
    warmup_target_1d = torch.from_numpy(wl_1d_norm[prefix_len:start_idx, :, 0]).unsqueeze(0).to(device)
    warmup_target_2d = torch.from_numpy(wl_2d_norm[prefix_len:start_idx, :, 0]).unsqueeze(0).to(device)

    # Extract warmup rainfall
    warmup_rainfall = input_2d[:, prefix_len:start_idx, :, 0:1]

    # Future rainfall for full horizon
    future_rainfall = input_2d[:, start_idx:, :, 0:1]

    if calibrate and warmup_len > 0:
        # Keep autograd enabled for latent optimization.
        c_e, z0_1d, z0_2d = calibrate_latents(
            model, graph,
            prefix_1d, prefix_2d,
            warmup_target_1d, warmup_target_2d,
            warmup_rainfall,
            num_steps=calibration_steps,
            lr=calibration_lr,
            device=device,
        )
        with torch.no_grad():
            outputs = model.forward_from_latents(
                graph, c_e, z0_1d, z0_2d,
                future_rainfall,
            )
    else:
        with torch.no_grad():
            outputs = model(
                graph, input_1d[:, :start_idx], input_2d[:, :start_idx],
                prefix_len=prefix_len,
                future_rainfall=future_rainfall,
            )

    pred_1d = outputs['pred_1d'][0, :horizon, :, 0].cpu().numpy()  # [horizon, N_1d]
    pred_2d = outputs['pred_2d'][0, :horizon, :, 0].cpu().numpy()  # [horizon, N_2d]

    # Denormalize predictions
    if dataset.norm_stats is not None:
        pred_1d = pred_1d * dataset.norm_stats['target_1d']['std'] + dataset.norm_stats['target_1d']['mean']
        pred_2d = pred_2d * dataset.norm_stats['target_2d']['std'] + dataset.norm_stats['target_2d']['mean']

    # Store predictions
    preds_1d[start_idx:start_idx + horizon] = pred_1d[:horizon]
    preds_2d[start_idx:start_idx + horizon] = pred_2d[:horizon]

    # Fill prefix and warmup with observed values
    preds_1d[:start_idx] = water_level_1d[:start_idx, :, 0]
    preds_2d[:start_idx] = water_level_2d[:start_idx, :, 0]

    return {
        'pred_1d': preds_1d,
        'pred_2d': preds_2d,
        'event_id': dataset.event_id,
    }


def format_submission(
    predictions: List[Dict],
    model_id: int,
    data_dir: str,
) -> pd.DataFrame:
    """
    Format predictions for Kaggle submission.

    Args:
        predictions: List of prediction dicts per event
        model_id: Model ID (1 or 2)
        data_dir: Data directory path

    Returns:
        DataFrame in submission format
    """
    rows = []

    for pred_dict in predictions:
        event_id = pred_dict['event_id']
        pred_1d = pred_dict['pred_1d']
        pred_2d = pred_dict['pred_2d']

        num_timesteps = pred_1d.shape[0]
        num_1d_nodes = pred_1d.shape[1]
        num_2d_nodes = pred_2d.shape[1]

        # Load node mapping to get original node IDs
        event_path = os.path.join(
            data_dir, f"Model_{model_id}", "test", f"event_{event_id}"
        )

        # Load original node data to get node IDs
        df_1d = pd.read_csv(os.path.join(event_path, "1d_nodes_dynamic_all.csv"))
        df_2d = pd.read_csv(os.path.join(event_path, "2d_nodes_dynamic_all.csv"))

        # Some dumps provide only node_idx; fall back cleanly.
        id_col_1d = 'node_id' if 'node_id' in df_1d.columns else 'node_idx'
        id_col_2d = 'node_id' if 'node_id' in df_2d.columns else 'node_idx'

        # Get unique node IDs (sorted by node_idx)
        node_ids_1d = (
            df_1d.sort_values(['timestep', 'node_idx'])
            .groupby('node_idx')[id_col_1d]
            .first()
            .values
        )
        node_ids_2d = (
            df_2d.sort_values(['timestep', 'node_idx'])
            .groupby('node_idx')[id_col_2d]
            .first()
            .values
        )

        # Create rows for 1D nodes
        for t in range(num_timesteps):
            for n, node_id in enumerate(node_ids_1d):
                rows.append({
                    'model_id': model_id,
                    'event_id': event_id,
                    'timestep': t,
                    'node_type': '1d',
                    'node_id': node_id,
                    'water_level': pred_1d[t, n],
                })

        # Create rows for 2D nodes
        for t in range(num_timesteps):
            for n, node_id in enumerate(node_ids_2d):
                rows.append({
                    'model_id': model_id,
                    'event_id': event_id,
                    'timestep': t,
                    'node_type': '2d',
                    'node_id': node_id,
                    'water_level': pred_2d[t, n],
                })

    df = pd.DataFrame(rows)
    return df


def main():
    args = parse_args()

    # Get device
    device = get_device(args.device)
    print(f"Using device: {device}")

    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model_state, hparams = load_model(args.checkpoint, device)

    # Build graph
    print(f"\nBuilding graph for Model {args.model_id}...")
    builder = FloodGraphBuilder(args.data_dir, args.model_id, add_knn_2d_edges=True, knn_k=8)
    graph = builder.build(split="test")
    graph = graph.to(device)

    # Get feature dimensions
    static_1d_dim = graph['1d'].x.shape[1]
    static_2d_dim = graph['2d'].x.shape[1]
    num_1d_nodes = graph['1d'].x.shape[0]
    num_2d_nodes = graph['2d'].x.shape[0]
    print(f"  1D nodes: {num_1d_nodes}")
    print(f"  2D nodes: {num_2d_nodes}")

    # Dynamic feature dimensions
    dynamic_1d_dim = 2
    dynamic_2d_dim = 3

    # Infer model config from state dict
    hidden_dim = hparams.get('hidden_dim', 64)
    latent_dim = hparams.get('latent_dim', 32)
    event_latent_dim = hparams.get('event_latent_dim', 16)

    # Try to infer from state dict if not in hparams
    for key, value in model_state.items():
        if 'spatial_encoder' in key and 'input_proj' in key and 'weight' in key:
            if '1d' in key:
                hidden_dim = value.shape[0]
                break

    # Create model
    print(f"\nCreating VGSSM model (hidden_dim={hidden_dim}, latent_dim={latent_dim})...")
    model = VGSSM(
        static_1d_dim=static_1d_dim,
        static_2d_dim=static_2d_dim,
        dynamic_1d_dim=dynamic_1d_dim,
        dynamic_2d_dim=dynamic_2d_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        event_latent_dim=event_latent_dim,
        num_gnn_layers=hparams.get('num_gnn_layers', 3),
        num_transition_gnn_layers=hparams.get('num_transition_gnn_layers', 2),
        num_heads=hparams.get('num_heads', 4),
        prediction_horizon=hparams.get('prediction_horizon', 90),
        use_event_latent=True,
        dropout=0.0,  # No dropout at inference
    )

    # Load weights
    model.load_state_dict(model_state, strict=False)
    model = model.to(device)
    model.eval()

    # Discover test events
    test_path = os.path.join(args.data_dir, f"Model_{args.model_id}", "test")
    test_events = []
    if os.path.exists(test_path):
        for item in os.listdir(test_path):
            if item.startswith("event_"):
                try:
                    event_id = int(item.split("_")[1])
                    test_events.append(event_id)
                except ValueError:
                    pass
    test_events = sorted(test_events)
    print(f"\nFound {len(test_events)} test events")

    # Compute normalization stats from training data (or use defaults)
    # For simplicity, we'll load a train event to get norm stats
    train_path = os.path.join(args.data_dir, f"Model_{args.model_id}", "train")
    train_events = []
    if os.path.exists(train_path):
        for item in os.listdir(train_path):
            if item.startswith("event_"):
                try:
                    event_id = int(item.split("_")[1])
                    train_events.append(event_id)
                except ValueError:
                    pass

    norm_stats = None
    if train_events:
        sample_ds = FloodEventDataset(
            args.data_dir, args.model_id, train_events[0], "train", graph,
            normalize=True,
        )
        norm_stats = sample_ds.norm_stats

    # Generate predictions
    print(f"\nGenerating predictions (calibrate={args.calibrate_latent})...")
    predictions = []

    for event_id in tqdm(test_events, desc="Events"):
        # Create dataset for this event
        dataset = FloodEventDataset(
            args.data_dir, args.model_id, event_id, "test", graph,
            seq_len=args.prefix_len,
            pred_len=1,
            normalize=True,
            normalization_stats=norm_stats,
        )

        # Generate predictions
        pred_dict = predict_event(
            model, graph, dataset,
            prefix_len=args.prefix_len,
            warmup_len=args.warmup_len,
            calibrate=args.calibrate_latent,
            calibration_steps=args.calibration_steps,
            calibration_lr=args.calibration_lr,
            device=device,
        )
        predictions.append(pred_dict)

    # Format submission
    print("\nFormatting submission...")
    df = format_submission(predictions, args.model_id, args.data_dir)

    # Save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    suffix = args.output_suffix if args.output_suffix else ''
    if args.calibrate_latent:
        suffix = f"_calibrated{suffix}"

    output_file = output_dir / f"submission_vgssm_model{args.model_id}{suffix}.parquet"
    df.to_parquet(output_file, index=False)
    print(f"\nSaved predictions to: {output_file}")
    print(f"  Total rows: {len(df):,}")

    # Also save as CSV for debugging
    csv_file = output_dir / f"submission_vgssm_model{args.model_id}{suffix}.csv"
    df.to_csv(csv_file, index=False)
    print(f"  CSV backup: {csv_file}")


if __name__ == '__main__':
    main()
