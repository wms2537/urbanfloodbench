#!/usr/bin/env python3
"""
Prediction script for Graph-TFT model.

Features:
- Multi-horizon prediction (all 90 steps at once)
- Optional test-time event latent calibration
- Proper denormalization

Usage:
    python predict_graph_tft.py --model_id 1 --checkpoint path/to/checkpoint.ckpt
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from src.data.graph_builder import FloodGraphBuilder
from src.data.dataset import FloodEventDataset, FloodDataModule
from src.models.graph_tft import GraphTFT


def load_model(checkpoint_path: str, model_config: Dict, device: torch.device) -> GraphTFT:
    """Load Graph-TFT model from checkpoint."""
    model = GraphTFT(**model_config)

    # Load state dict
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Handle Lightning checkpoint format
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        # Remove 'model.' prefix if present
        state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
    else:
        state_dict = checkpoint

    # CRITICAL: Use strict=False but CHECK for mismatches
    incompat = model.load_state_dict(state_dict, strict=False)

    if incompat.missing_keys:
        print(f"WARNING: Missing keys in checkpoint: {len(incompat.missing_keys)}")
        for k in incompat.missing_keys[:5]:
            print(f"  - {k}")
        if len(incompat.missing_keys) > 5:
            print(f"  ... and {len(incompat.missing_keys) - 5} more")

    if incompat.unexpected_keys:
        print(f"WARNING: Unexpected keys in checkpoint: {len(incompat.unexpected_keys)}")
        for k in incompat.unexpected_keys[:5]:
            print(f"  - {k}")
        if len(incompat.unexpected_keys) > 5:
            print(f"  ... and {len(incompat.unexpected_keys) - 5} more")

    # FAIL HARD if there are missing keys (model won't work correctly)
    if incompat.missing_keys:
        raise RuntimeError(
            f"Checkpoint/model config mismatch! {len(incompat.missing_keys)} missing keys. "
            f"Check hidden_dim, num_gnn_layers, num_tft_layers match training config. "
            f"Model 1 should use hidden_dim=64, Model 2 should use hidden_dim=96 (or your training values)."
        )

    model = model.to(device)
    model.eval()

    return model


def predict_event(
    model: GraphTFT,
    graph: HeteroData,
    event_data_1d: torch.Tensor,
    event_data_2d: torch.Tensor,
    warmup_steps: int = 10,
    norm_stats: Optional[Dict] = None,
    calibrate_latent: bool = False,
    calibration_steps: int = 30,
    device: torch.device = torch.device('cpu'),
) -> Dict[str, torch.Tensor]:
    """
    Generate predictions for a single event.

    Args:
        model: Graph-TFT model
        graph: Heterogeneous graph
        event_data_1d: Full event 1D data [timesteps, num_1d_nodes, features]
        event_data_2d: Full event 2D data [timesteps, num_2d_nodes, features]
        warmup_steps: Number of warmup timesteps (ground truth available)
        norm_stats: Normalization statistics
        calibrate_latent: Whether to do test-time calibration
        calibration_steps: Number of calibration optimization steps
        device: Device to use

    Returns:
        Dict with predictions
    """
    num_timesteps = event_data_1d.shape[0]
    horizon = num_timesteps - warmup_steps

    # Move event data to device first
    event_data_1d = event_data_1d.to(device)
    event_data_2d = event_data_2d.to(device)

    # Normalize input
    if norm_stats is not None:
        input_1d = (event_data_1d - norm_stats['1d']['mean'].to(device)) / norm_stats['1d']['std'].to(device)
        input_2d = (event_data_2d - norm_stats['2d']['mean'].to(device)) / norm_stats['2d']['std'].to(device)
    else:
        input_1d = event_data_1d
        input_2d = event_data_2d

    # Add batch dimension
    input_1d = input_1d[:warmup_steps].unsqueeze(0).to(device)  # [1, warmup, nodes, features]
    input_2d = input_2d[:warmup_steps].unsqueeze(0).to(device)

    # Get future rainfall (known)
    future_rainfall = event_data_2d[warmup_steps:, :, 0:1]  # [horizon, nodes, 1]
    if norm_stats is not None:
        future_rainfall = (future_rainfall - norm_stats['2d']['mean'][0].to(device)) / norm_stats['2d']['std'][0].to(device)
    future_rainfall = future_rainfall.unsqueeze(0).to(device)  # [1, horizon, nodes, 1]

    # Optional: Test-time calibration of event latent (needs gradients, so outside no_grad)
    if calibrate_latent:
        # Use warmup period for calibration
        target_1d = event_data_1d[:warmup_steps, :, 0]  # Water level
        target_2d = event_data_2d[:warmup_steps, :, 1]  # Water level (after rainfall)

        if norm_stats is not None:
            target_1d = (target_1d - norm_stats['target_1d']['mean'].to(device)) / norm_stats['target_1d']['std'].to(device)
            target_2d = (target_2d - norm_stats['target_2d']['mean'].to(device)) / norm_stats['target_2d']['std'].to(device)

        target_1d = target_1d.to(device)
        target_2d = target_2d.to(device)

        # Get rainfall during warmup period for calibration
        rainfall_prefix = event_data_2d[:warmup_steps, :, 0:1]  # [warmup, nodes, 1]
        if norm_stats is not None:
            rainfall_prefix = (rainfall_prefix - norm_stats['2d']['mean'][0].to(device)) / norm_stats['2d']['std'][0].to(device)
        rainfall_prefix = rainfall_prefix.unsqueeze(0).to(device)  # [1, warmup, nodes, 1]

        # Optimize c_e (this requires gradients)
        c_e = model.optimize_event_latent(
            graph, input_1d, input_2d,
            target_1d, target_2d,
            rainfall_prefix=rainfall_prefix,
            num_steps=calibration_steps,
        )
    else:
        c_e = None

    with torch.no_grad():
        # Multi-horizon prediction
        # Adjust model's prediction horizon if needed
        if horizon > model.prediction_horizon:
            # Need to do chunked prediction or fall back to autoregressive
            outputs = model.forward_autoregressive(
                graph, input_1d, input_2d,
                horizon=horizon,
                prefix_len=warmup_steps,
                full_rainfall=torch.cat([input_2d[..., 0:1], future_rainfall], dim=1),
            )
        else:
            outputs = model(
                graph, input_1d, input_2d,
                prefix_len=warmup_steps,
                future_rainfall=future_rainfall[:, :horizon],
                c_e_override=c_e,  # Use calibrated event latent if available
            )

    # Get predictions (water level is first output feature)
    pred_1d = outputs['pred_1d'][0, :horizon, :, 0]  # [horizon, num_1d_nodes]
    pred_2d = outputs['pred_2d'][0, :horizon, :, 0]  # [horizon, num_2d_nodes]

    # Denormalize
    if norm_stats is not None:
        pred_1d = pred_1d * norm_stats['target_1d']['std'].to(device) + norm_stats['target_1d']['mean'].to(device)
        pred_2d = pred_2d * norm_stats['target_2d']['std'].to(device) + norm_stats['target_2d']['mean'].to(device)

    # Prepend warmup ground truth (for full sequence)
    warmup_1d = event_data_1d[:warmup_steps, :, 0].to(device)  # Water level
    warmup_2d = event_data_2d[:warmup_steps, :, 1].to(device)  # Water level

    full_pred_1d = torch.cat([warmup_1d, pred_1d], dim=0)
    full_pred_2d = torch.cat([warmup_2d, pred_2d], dim=0)

    return {
        'pred_1d': full_pred_1d.cpu(),
        'pred_2d': full_pred_2d.cpu(),
    }


def main():
    parser = argparse.ArgumentParser(description='Generate predictions with Graph-TFT')

    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--model_id', type=int, required=True, choices=[1, 2])
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output', type=str, default=None)

    # Model config (should match training)
    # Default values for Model 1; will be adjusted for Model 2
    parser.add_argument('--hidden_dim', type=int, default=None,
                        help='Hidden dimension (default: 64 for Model 1, 96 for Model 2)')
    parser.add_argument('--event_latent_dim', type=int, default=16)
    parser.add_argument('--num_gnn_layers', type=int, default=None,
                        help='Number of GNN layers (default: 3 for Model 1, 4 for Model 2)')
    parser.add_argument('--num_tft_layers', type=int, default=None,
                        help='Number of TFT layers (default: 2 for Model 1, 3 for Model 2)')
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--prediction_horizon', type=int, default=90)
    parser.add_argument('--dropout', type=float, default=None,
                        help='Dropout rate (default: 0.2 for Model 1, 0.15 for Model 2)')

    # Prediction options
    parser.add_argument('--warmup_steps', type=int, default=10)
    parser.add_argument('--calibrate_latent', action='store_true',
                        help='Do test-time event latent calibration')
    parser.add_argument('--calibration_steps', type=int, default=30)

    parser.add_argument('--device', type=str, default='auto')

    args = parser.parse_args()

    # Set device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Set model-specific defaults
    if args.model_id == 2:
        # Model 2: Larger graph needs larger capacity
        hidden_dim = args.hidden_dim or 96
        num_gnn_layers = args.num_gnn_layers or 4
        num_tft_layers = args.num_tft_layers or 3
        dropout = args.dropout or 0.15
    else:
        # Model 1: Default settings
        hidden_dim = args.hidden_dim or 64
        num_gnn_layers = args.num_gnn_layers or 3
        num_tft_layers = args.num_tft_layers or 2
        dropout = args.dropout or 0.2

    print(f"Model config: hidden_dim={hidden_dim}, gnn_layers={num_gnn_layers}, tft_layers={num_tft_layers}")

    # Build graph
    print(f"Building graph for model {args.model_id}...")
    graph_builder = FloodGraphBuilder(args.data_dir, args.model_id, add_knn_2d_edges=True, knn_k=8)
    graph = graph_builder.build(split="test")
    graph = graph.to(device)

    # Get dimensions
    static_1d_dim = graph['1d'].x.shape[0]
    static_2d_dim = graph['2d'].x.shape[0]
    num_1d_nodes = graph['1d'].x.shape[0]
    num_2d_nodes = graph['2d'].x.shape[0]

    # Model config
    model_config = {
        'static_1d_dim': graph['1d'].x.shape[1],
        'static_2d_dim': graph['2d'].x.shape[1],
        'dynamic_1d_dim': 2,
        'dynamic_2d_dim': 3,
        'hidden_dim': hidden_dim,
        'event_latent_dim': args.event_latent_dim,
        'num_gnn_layers': num_gnn_layers,
        'num_tft_layers': num_tft_layers,
        'num_heads': args.num_heads,
        'prediction_horizon': args.prediction_horizon,
        'dropout': dropout,
    }

    # Load model
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, model_config, device)

    # Get normalization stats from training data using FloodDataModule
    print("Computing normalization stats from training data...")
    data_module = FloodDataModule(
        data_dir=args.data_dir,
        model_id=args.model_id,
        batch_size=1,
        seq_len=10,
        pred_len=args.prediction_horizon,
        stride=1,
        num_workers=0,
        graph_kwargs={'add_knn_2d_edges': True, 'knn_k': 8},
    )
    data_module.setup('fit')
    norm_stats = data_module.norm_stats

    # Convert to tensors
    for key in norm_stats:
        if isinstance(norm_stats[key], dict):
            for subkey in norm_stats[key]:
                if isinstance(norm_stats[key][subkey], (np.ndarray, list)):
                    norm_stats[key][subkey] = torch.tensor(norm_stats[key][subkey], dtype=torch.float32)
                elif not isinstance(norm_stats[key][subkey], torch.Tensor):
                    norm_stats[key][subkey] = torch.tensor(norm_stats[key][subkey], dtype=torch.float32)

    # Get test events
    test_events_path = Path(args.data_dir) / f"Model_{args.model_id}" / "test_events.txt"
    if test_events_path.exists():
        with open(test_events_path) as f:
            test_events = [line.strip() for line in f if line.strip()]
    else:
        # Fallback: find event directories (extract numeric IDs)
        test_dir = Path(args.data_dir) / f"Model_{args.model_id}" / "test"
        test_events = sorted([
            int(d.name.replace('event_', ''))
            for d in test_dir.iterdir()
            if d.is_dir() and d.name.startswith('event_')
        ])

    print(f"Found {len(test_events)} test events")

    # Generate predictions
    all_results = []

    for event_id in tqdm(test_events, desc="Predicting"):
        # Load event data
        event_dataset = FloodEventDataset(
            args.data_dir, args.model_id, event_id, "test",
            graph, seq_len=1, pred_len=1, stride=1,
            normalize=False,  # We'll normalize manually
        )

        # Get full event data
        event_data_1d = event_dataset.dynamic_1d  # [timesteps, nodes, features]
        event_data_2d = event_dataset.dynamic_2d

        event_data_1d = torch.tensor(event_data_1d, dtype=torch.float32)
        event_data_2d = torch.tensor(event_data_2d, dtype=torch.float32)

        # Predict
        preds = predict_event(
            model, graph,
            event_data_1d, event_data_2d,
            warmup_steps=args.warmup_steps,
            norm_stats=norm_stats,
            calibrate_latent=args.calibrate_latent,
            calibration_steps=args.calibration_steps,
            device=device,
        )

        # Format results
        num_timesteps = event_data_1d.shape[0]

        # 1D predictions
        for t in range(num_timesteps):
            for node_idx in range(num_1d_nodes):
                all_results.append({
                    'model_id': args.model_id,
                    'event_id': event_id,
                    'element_type': '1d',
                    'element_id': node_idx,
                    'timestep': t,
                    'water_level': preds['pred_1d'][t, node_idx].item(),
                })

        # 2D predictions
        for t in range(num_timesteps):
            for node_idx in range(num_2d_nodes):
                all_results.append({
                    'model_id': args.model_id,
                    'event_id': event_id,
                    'element_type': '2d',
                    'element_id': node_idx,
                    'timestep': t,
                    'water_level': preds['pred_2d'][t, node_idx].item(),
                })

    # Create DataFrame
    df = pd.DataFrame(all_results)

    # Set output path
    if args.output is None:
        output_path = f"submission_graph_tft_model{args.model_id}.parquet"
    else:
        output_path = args.output

    # Save
    df.to_parquet(output_path, index=False)
    print(f"Saved predictions to {output_path}")
    print(f"Shape: {df.shape}")
    print(f"Water level stats:\n{df['water_level'].describe()}")


if __name__ == '__main__':
    main()
