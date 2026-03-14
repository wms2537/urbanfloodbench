"""
Prediction and Kaggle submission script for CL-DTS.
"""

import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Tuple

from src.data.graph_builder import FloodGraphBuilder
from src.data.dataset import FloodEventDataset
from src.models.cldts import CLDTS
from src.training.trainer import FloodTrainer


def compute_global_norm_stats(data_dir: str, model_id: int, graph) -> Dict:
    """Compute normalization stats from TRAINING data (not test data).

    This is critical - the model was trained with these stats, so inference
    must use the same normalization.
    """
    train_path = os.path.join(data_dir, f"Model_{model_id}", "train")
    train_events = [int(d.split("_")[1]) for d in os.listdir(train_path) if d.startswith("event_")]
    train_events = sorted(train_events)

    all_1d = []
    all_2d = []
    all_wl_1d = []
    all_wl_2d = []

    print(f"  Computing normalization stats from {len(train_events)} training events...")
    for event_id in train_events:
        ds = FloodEventDataset(
            data_dir, model_id, event_id, "train",
            graph, seq_len=16, pred_len=1, stride=1,
            normalize=False  # Get raw data
        )
        all_1d.append(ds.dynamic_1d)
        all_2d.append(ds.dynamic_2d)
        all_wl_1d.append(ds.water_level_1d)
        all_wl_2d.append(ds.water_level_2d)

    all_1d = np.concatenate(all_1d, axis=0)
    all_2d = np.concatenate(all_2d, axis=0)
    all_wl_1d = np.concatenate(all_wl_1d, axis=0)
    all_wl_2d = np.concatenate(all_wl_2d, axis=0)

    return {
        '1d': {
            'mean': all_1d.mean(axis=(0, 1)),
            'std': all_1d.std(axis=(0, 1)) + 1e-8
        },
        '2d': {
            'mean': all_2d.mean(axis=(0, 1)),
            'std': all_2d.std(axis=(0, 1)) + 1e-8
        },
        'target_1d': {
            'mean': all_wl_1d.mean(),
            'std': all_wl_1d.std() + 1e-8
        },
        'target_2d': {
            'mean': all_wl_2d.mean(),
            'std': all_wl_2d.std() + 1e-8
        }
    }


def load_model(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[FloodTrainer, Dict]:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    hparams = checkpoint['hyper_parameters']

    # Create model
    model = FloodTrainer(**hparams)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.to(device)

    return model, hparams


def predict_event(
    model: FloodTrainer,
    graph,
    event_dataset: FloodEventDataset,
    device: torch.device,
    warmup_steps: int = 10,
) -> Dict[str, np.ndarray]:
    """
    Predict water levels for a single event.

    The submission expects predictions starting after warmup_steps timesteps.
    Uses autoregressive rollout from warmup prefix.

    Args:
        model: Trained model
        graph: Graph structure
        event_dataset: Dataset for the event
        device: Computation device
        warmup_steps: Number of warmup timesteps to skip (use as input)

    Returns:
        Dict with 'water_level_1d' and 'water_level_2d' predictions
        Shape: [num_pred_timesteps, num_nodes]
    """
    # Get full sequence
    full_seq = event_dataset.get_full_sequence()
    num_timesteps = full_seq['num_timesteps']

    # Predictions start after warmup
    num_pred_timesteps = num_timesteps - warmup_steps

    input_1d = full_seq['input_1d'].unsqueeze(0).to(device)  # [1, T, N1, F]
    input_2d = full_seq['input_2d'].unsqueeze(0).to(device)  # [1, T, N2, F]

    # Set graph
    model.set_graph(graph)
    graph = graph.to(device)

    # Use warmup sequence as input for autoregressive prediction
    # We predict all timesteps after warmup
    model.eval()

    with torch.no_grad():
        # Get initial spatial encoding
        spatial_1d, spatial_2d = model.model.encode_spatial(graph)

        # Encode event latent from warmup prefix
        prefix_1d = input_1d[:, :warmup_steps]
        prefix_2d = input_2d[:, :warmup_steps]
        c_e, _, _ = model.model.encode_event_latent(prefix_1d, prefix_2d)

        # Initialize hidden states
        h_1d = None
        h_2d = None

        # Process warmup sequence to initialize hidden states
        st_out_1d, h_1d = model.model.st_encoder_1d(spatial_1d, prefix_1d, h_1d)
        st_out_2d, h_2d = model.model.st_encoder_2d(spatial_2d, prefix_2d, h_2d)

        # Get current state from last warmup step
        curr_z_1d = st_out_1d[:, -1:]  # [1, 1, N1, hidden]
        curr_z_2d = st_out_2d[:, -1:]  # [1, 1, N2, hidden]

        # Use last warmup input for autoregressive prediction
        curr_input_1d = input_1d[:, warmup_steps-1:warmup_steps]  # [1, 1, N1, F]
        curr_input_2d = input_2d[:, warmup_steps-1:warmup_steps]  # [1, 1, N2, F]

        all_pred_wl_1d = []  # water_level predictions only (for submission)
        all_pred_wl_2d = []

        # Get feature dimensions
        d1d = curr_input_1d.shape[-1]  # 1D feature dimension
        d2d = curr_input_2d.shape[-1]  # 2D feature dimension

        # TRUE AUTOREGRESSIVE prediction with multi-output decoder
        for t in range(num_pred_timesteps):
            # Decode ALL features
            # pred_1d: [1, 1, N1, output_dim_1d] (water_level + inlet_flow)
            # pred_2d: [1, 1, N2, output_dim_2d] (water_level + water_volume)
            pred_1d = model.model.decoder_1d(curr_z_1d, spatial_1d, c_e)
            pred_2d = model.model.decoder_2d(curr_z_2d, spatial_2d, c_e)

            pred_1d_squeezed = pred_1d.squeeze(1)  # [1, N1, output_dim]
            pred_2d_squeezed = pred_2d.squeeze(1)  # [1, N2, output_dim]

            # Store water_level predictions only (for submission)
            all_pred_wl_1d.append(pred_1d_squeezed[:, :, 0:1].cpu())  # [1, N1, 1]
            all_pred_wl_2d.append(pred_2d_squeezed[:, :, 0:1].cpu())  # [1, N2, 1]

            # Prepare next input - TRUE AUTOREGRESSIVE (use ALL predicted features)
            actual_timestep = warmup_steps + t

            # Create next input tensors
            next_input_1d = torch.zeros(1, 1, curr_input_1d.shape[2], d1d, device=device)
            next_input_2d = torch.zeros(1, 1, curr_input_2d.shape[2], d2d, device=device)

            # 1D: use all predictions [water_level, inlet_flow]
            next_input_1d[:, 0, :, :] = pred_1d_squeezed

            # 2D: use actual rainfall + predicted [water_level, water_volume]
            if actual_timestep < num_timesteps:
                next_input_2d[:, 0, :, 0] = input_2d[:, actual_timestep, :, 0]  # actual rainfall
            else:
                next_input_2d[:, 0, :, 0] = curr_input_2d[:, 0, :, 0]  # keep last rainfall
            next_input_2d[:, 0, :, 1:] = pred_2d_squeezed  # predicted water_level + water_volume

            # Process through temporal encoder (single pass - fixes double encoder bug)
            curr_z_1d, h_1d = model.model.st_encoder_1d(spatial_1d, next_input_1d, h_1d)
            curr_z_2d, h_2d = model.model.st_encoder_2d(spatial_2d, next_input_2d, h_2d)
            curr_z_1d = curr_z_1d[:, -1:]
            curr_z_2d = curr_z_2d[:, -1:]
            curr_input_1d = next_input_1d
            curr_input_2d = next_input_2d

    # Stack water_level predictions: [num_pred_timesteps, num_nodes]
    pred_1d = torch.cat(all_pred_wl_1d, dim=0).squeeze(-1).numpy()  # [T, N1]
    pred_2d = torch.cat(all_pred_wl_2d, dim=0).squeeze(-1).numpy()  # [T, N2]

    # Denormalize if normalization stats available
    if event_dataset.norm_stats is not None:
        pred_1d = pred_1d * event_dataset.norm_stats['target_1d']['std'] + event_dataset.norm_stats['target_1d']['mean']
        pred_2d = pred_2d * event_dataset.norm_stats['target_2d']['std'] + event_dataset.norm_stats['target_2d']['mean']

    # Ensure non-negative water levels
    pred_1d = np.maximum(pred_1d, 0.0)
    pred_2d = np.maximum(pred_2d, 0.0)

    return {
        'water_level_1d': pred_1d,  # [num_pred_timesteps, N1]
        'water_level_2d': pred_2d,  # [num_pred_timesteps, N2]
        'num_pred_timesteps': num_pred_timesteps,
    }


def create_submission(
    predictions: Dict[int, Dict[int, Dict[str, np.ndarray]]],
    sample_submission_path: str,
    output_path: str,
):
    """
    Create Kaggle submission file.

    Args:
        predictions: Dict[model_id][event_id] = {
            'water_level_1d': [T, N1],
            'water_level_2d': [T, N2],
            'num_pred_timesteps': int
        }
        sample_submission_path: Path to sample submission parquet
        output_path: Output path for submission file
    """
    print("Loading sample submission...")
    sub = pd.read_parquet(sample_submission_path)

    print(f"Sample submission shape: {sub.shape}")
    print(f"Columns: {sub.columns.tolist()}")

    # Group submission by model_id and event_id for efficient filling
    # The submission rows are ordered by timestep for each (model_id, event_id, node_type, node_id)
    print("Filling predictions...")

    water_levels = np.zeros(len(sub), dtype=np.float32)

    for model_id in predictions:
        for event_id in predictions[model_id]:
            pred = predictions[model_id][event_id]
            pred_1d = pred['water_level_1d']  # [T, N1]
            pred_2d = pred['water_level_2d']  # [T, N2]

            # Get indices for this event
            mask_event = (sub['model_id'] == model_id) & (sub['event_id'] == event_id)

            # Process 1D nodes
            mask_1d = mask_event & (sub['node_type'] == 1)
            if mask_1d.any():
                indices_1d = sub.index[mask_1d].values
                rows_1d = sub.loc[mask_1d]

                # Group by node_id
                for node_id in rows_1d['node_id'].unique():
                    if node_id >= pred_1d.shape[1]:
                        continue
                    node_mask = rows_1d['node_id'] == node_id
                    node_indices = indices_1d[node_mask.values]

                    # The rows for this node are sequential timesteps
                    num_timesteps = len(node_indices)
                    if num_timesteps <= pred_1d.shape[0]:
                        water_levels[node_indices] = pred_1d[:num_timesteps, node_id]
                    else:
                        # Pad with last value if needed
                        water_levels[node_indices[:pred_1d.shape[0]]] = pred_1d[:, node_id]
                        water_levels[node_indices[pred_1d.shape[0]:]] = pred_1d[-1, node_id]

            # Process 2D nodes
            mask_2d = mask_event & (sub['node_type'] == 2)
            if mask_2d.any():
                indices_2d = sub.index[mask_2d].values
                rows_2d = sub.loc[mask_2d]

                for node_id in rows_2d['node_id'].unique():
                    if node_id >= pred_2d.shape[1]:
                        continue
                    node_mask = rows_2d['node_id'] == node_id
                    node_indices = indices_2d[node_mask.values]

                    num_timesteps = len(node_indices)
                    if num_timesteps <= pred_2d.shape[0]:
                        water_levels[node_indices] = pred_2d[:num_timesteps, node_id]
                    else:
                        water_levels[node_indices[:pred_2d.shape[0]]] = pred_2d[:, node_id]
                        water_levels[node_indices[pred_2d.shape[0]:]] = pred_2d[-1, node_id]

    sub['water_level'] = water_levels

    # Save
    print(f"Saving submission to {output_path}...")
    if output_path.endswith('.parquet'):
        sub.to_parquet(output_path, index=False)
    else:
        sub.to_csv(output_path, index=False)

    print("Done!")
    return sub


def main():
    parser = argparse.ArgumentParser(description="Generate predictions for Kaggle submission")

    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--checkpoint_1", type=str, default=None, help="Model 1 checkpoint path")
    parser.add_argument("--checkpoint_2", type=str, default=None, help="Model 2 checkpoint path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Single checkpoint for both models")
    parser.add_argument("--sample_submission", type=str, default="data/sample_submission.parquet",
                        help="Sample submission file")
    parser.add_argument("--output", type=str, default="submission.parquet", help="Output submission file")
    parser.add_argument("--model_id", type=int, default=None, help="Specific model ID (None = all)")
    parser.add_argument("--warmup_steps", type=int, default=10, help="Number of warmup timesteps")
    parser.add_argument("--device", type=str, default="auto", help="Device (auto, cpu, cuda, mps)")

    args = parser.parse_args()

    # Set device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Determine which models to process
    model_ids = [args.model_id] if args.model_id else [1, 2]

    # Collect predictions
    all_predictions = {}

    for model_id in model_ids:
        print(f"\n{'='*50}")
        print(f"Processing Model {model_id}...")
        print(f"{'='*50}")

        # Get checkpoint for this model
        if model_id == 1 and args.checkpoint_1:
            checkpoint_path = args.checkpoint_1
        elif model_id == 2 and args.checkpoint_2:
            checkpoint_path = args.checkpoint_2
        elif args.checkpoint:
            checkpoint_path = args.checkpoint
        else:
            print(f"  No checkpoint provided for model {model_id}, skipping...")
            continue

        # Load model
        print(f"Loading model from {checkpoint_path}...")
        model, hparams = load_model(checkpoint_path, device)

        # Build graph
        print("Building graph...")
        builder = FloodGraphBuilder(args.data_dir, model_id)
        graph = builder.build(split="test")
        print(f"Graph: 1D nodes={graph['1d'].x.shape[0]}, 2D nodes={graph['2d'].x.shape[0]}")

        # CRITICAL: Compute normalization stats from TRAINING data, not test data
        # This ensures consistency with how the model was trained
        global_norm_stats = compute_global_norm_stats(args.data_dir, model_id, graph)
        print(f"  Normalization stats computed from training data:")
        print(f"    target_1d: mean={global_norm_stats['target_1d']['mean']:.4f}, std={global_norm_stats['target_1d']['std']:.4f}")
        print(f"    target_2d: mean={global_norm_stats['target_2d']['mean']:.4f}, std={global_norm_stats['target_2d']['std']:.4f}")

        # Discover test events
        test_path = os.path.join(args.data_dir, f"Model_{model_id}", "test")
        test_events = [int(d.split("_")[1]) for d in os.listdir(test_path) if d.startswith("event_")]
        test_events = sorted(test_events)

        print(f"Found {len(test_events)} test events")

        all_predictions[model_id] = {}

        for event_id in tqdm(test_events, desc=f"Model {model_id} events"):
            # Create dataset for event with TRAINING normalization stats
            event_dataset = FloodEventDataset(
                args.data_dir, model_id, event_id, "test",
                graph, seq_len=16, pred_len=1, stride=1,
                normalize=True, normalization_stats=global_norm_stats
            )

            # Predict
            try:
                predictions = predict_event(
                    model, graph, event_dataset, device,
                    warmup_steps=args.warmup_steps
                )
                all_predictions[model_id][event_id] = predictions
            except Exception as e:
                print(f"  Error on event {event_id}: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Create submission
    if all_predictions:
        create_submission(
            all_predictions,
            args.sample_submission,
            args.output,
        )
    else:
        print("No predictions generated!")


if __name__ == "__main__":
    main()
