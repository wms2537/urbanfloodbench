#!/usr/bin/env python3
"""
Training script for VGSSM (Variational Graph State-Space Model).

VGSSM extends Graph-TFT with per-timestep latent dynamics (z_t) that model
evolving hydraulic states in addition to the event latent (c_e).

Usage:
    python train_vgssm.py --model_id 1 --exp_name vgssm_v1 --max_epochs 30
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.dataset import FloodDataModule
from src.data.graph_builder import FloodGraphBuilder
from src.models.vgssm import VGSSM
from src.training.vgssm_trainer import VGSSMTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train VGSSM model')

    # Data
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to data directory')
    parser.add_argument('--model_id', type=int, required=True, choices=[1, 2],
                        help='Model ID (1 or 2)')

    # Model architecture
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension')
    parser.add_argument('--latent_dim', type=int, default=32,
                        help='Per-timestep latent dimension (z_t)')
    parser.add_argument('--event_latent_dim', type=int, default=16,
                        help='Event latent dimension (c_e)')
    parser.add_argument('--num_gnn_layers', type=int, default=3,
                        help='Number of GNN layers for spatial encoder')
    parser.add_argument('--num_transition_gnn_layers', type=int, default=2,
                        help='Number of GNN layers in latent transition')
    parser.add_argument('--num_heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')

    # Multi-horizon prediction
    parser.add_argument('--prediction_horizon', type=int, default=90,
                        help='Number of future steps to predict')
    parser.add_argument('--prefix_len', type=int, default=10,
                        help='Prefix length for encoding')

    # Training
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--max_epochs', type=int, default=30,
                        help='Maximum epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--beta_ce', type=float, default=0.01,
                        help='KL loss weight for event latent c_e')
    parser.add_argument('--beta_z', type=float, default=0.001,
                        help='KL loss weight for initial state z_0')
    parser.add_argument('--horizon_weighting', type=str, default='linear',
                        choices=['uniform', 'linear', 'exp'],
                        help='Horizon weighting strategy')
    parser.add_argument('--loss_type', type=str, default='mse',
                        choices=['mse', 'huber'],
                        help='Loss function type')
    parser.add_argument('--huber_delta', type=float, default=1.0,
                        help='Delta for Huber loss')
    parser.add_argument('--free_bits_ce', type=float, default=0.1,
                        help='Free bits for c_e KL (prevents collapse)')
    parser.add_argument('--free_bits_z', type=float, default=0.05,
                        help='Free bits for z_0 KL (prevents collapse)')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='Epochs for KL annealing warmup')

    # Data loading
    parser.add_argument('--seq_len', type=int, default=10,
                        help='Input sequence length (prefix)')
    parser.add_argument('--stride', type=int, default=4,
                        help='Stride for data sampling')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    # Experiment
    parser.add_argument('--exp_name', type=str, default='vgssm',
                        help='Experiment name')
    parser.add_argument('--patience', type=int, default=7,
                        help='Early stopping patience')
    parser.add_argument('--accumulate_grad_batches', type=int, default=1,
                        help='Gradient accumulation steps')

    # Hardware
    parser.add_argument('--accelerator', type=str, default='auto',
                        help='Accelerator (auto, gpu, mps, cpu)')
    parser.add_argument('--devices', type=int, default=1,
                        help='Number of devices')

    return parser.parse_args()


def get_model_specific_config(model_id: int, args) -> dict:
    """
    Get model-specific hyperparameters.
    Model 2 has 12x more 1D nodes and needs larger capacity.
    """
    if model_id == 2:
        # Model 2: Larger graph (198 1D nodes vs 17)
        return {
            'hidden_dim': max(args.hidden_dim, 96),
            'latent_dim': max(args.latent_dim, 48),  # Larger latent for more nodes
            'num_gnn_layers': max(args.num_gnn_layers, 4),
            'num_transition_gnn_layers': max(args.num_transition_gnn_layers, 3),
            'dropout': min(args.dropout, 0.15),
        }
    else:
        # Model 1: Default settings
        return {
            'hidden_dim': args.hidden_dim,
            'latent_dim': args.latent_dim,
            'num_gnn_layers': args.num_gnn_layers,
            'num_transition_gnn_layers': args.num_transition_gnn_layers,
            'dropout': args.dropout,
        }


def main():
    args = parse_args()

    # Set random seed
    pl.seed_everything(42, workers=True)

    # Get model-specific config
    model_config = get_model_specific_config(args.model_id, args)

    print("=" * 60)
    print(f"Training VGSSM for Model {args.model_id}")
    print(f"Experiment: {args.exp_name}")
    print(f"Model-specific config:")
    print(f"  hidden_dim={model_config['hidden_dim']}")
    print(f"  latent_dim={model_config['latent_dim']}")
    print(f"  gnn_layers={model_config['num_gnn_layers']}")
    print(f"  transition_gnn_layers={model_config['num_transition_gnn_layers']}")
    print(f"  beta_ce={args.beta_ce}, beta_z={args.beta_z}")
    print("=" * 60)

    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        print("MPS (Apple Silicon) available")
    else:
        print("Running on CPU")

    # Create data module
    print("\n[1/4] Loading data...")
    data_module = FloodDataModule(
        data_dir=args.data_dir,
        model_id=args.model_id,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        pred_len=args.prediction_horizon,
        stride=args.stride,
        num_workers=args.num_workers,
        graph_kwargs={'add_knn_2d_edges': True, 'knn_k': 8},
    )
    data_module.setup('fit')

    # Get normalization stats and graph
    norm_stats = data_module.norm_stats
    graph = data_module.graph
    print(f"  Training samples: {sum(len(ds) for ds in data_module.train_datasets)}")
    print(f"  Validation samples: {sum(len(ds) for ds in data_module.val_datasets)}")

    # Get feature dimensions
    print("\n[2/4] Building heterogeneous graph...")
    static_1d_dim = graph['1d'].x.shape[1]
    static_2d_dim = graph['2d'].x.shape[1]
    num_1d_nodes = graph['1d'].x.shape[0]
    num_2d_nodes = graph['2d'].x.shape[0]
    print(f"  1D nodes: {num_1d_nodes}, features: {static_1d_dim}")
    print(f"  2D nodes: {num_2d_nodes}, features: {static_2d_dim}")

    # Dynamic feature dimensions
    dynamic_1d_dim = 2  # [water_level, inlet_flow]
    dynamic_2d_dim = 3  # [rainfall, water_level, water_volume]

    # Move graph to device
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    graph = graph.to(device)

    # Create VGSSM model
    print("\n[3/4] Creating VGSSM model...")
    model = VGSSM(
        static_1d_dim=static_1d_dim,
        static_2d_dim=static_2d_dim,
        dynamic_1d_dim=dynamic_1d_dim,
        dynamic_2d_dim=dynamic_2d_dim,
        hidden_dim=model_config['hidden_dim'],
        latent_dim=model_config['latent_dim'],
        event_latent_dim=args.event_latent_dim,
        num_gnn_layers=model_config['num_gnn_layers'],
        num_transition_gnn_layers=model_config['num_transition_gnn_layers'],
        num_heads=args.num_heads,
        prediction_horizon=args.prediction_horizon,
        use_event_latent=True,
        dropout=model_config['dropout'],
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {num_params:,}")
    print(f"  Trainable parameters: {num_trainable:,}")

    # Parameter breakdown
    spatial_params = sum(p.numel() for p in model.spatial_encoder.parameters())
    transition_params = sum(p.numel() for p in model.transition.parameters())
    decoder_params = sum(p.numel() for p in model.decoder_1d.parameters()) + sum(p.numel() for p in model.decoder_2d.parameters())
    inference_params = sum(p.numel() for p in model.z0_encoder_1d.parameters()) + sum(p.numel() for p in model.z0_encoder_2d.parameters())
    print(f"  - Spatial encoder: {spatial_params:,}")
    print(f"  - Latent transition: {transition_params:,}")
    print(f"  - Latent decoders: {decoder_params:,}")
    print(f"  - Inference nets: {inference_params:,}")

    # Create trainer module
    trainer_module = VGSSMTrainer(
        model=model,
        graph=graph,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        beta_ce=args.beta_ce,
        beta_z=args.beta_z,
        horizon_weighting=args.horizon_weighting,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.max_epochs,
        norm_stats=norm_stats,
        loss_type=args.loss_type,
        huber_delta=args.huber_delta,
        free_bits_ce=args.free_bits_ce,
        free_bits_z=args.free_bits_z,
    )

    # Callbacks
    checkpoint_dir = Path('checkpoints') / f'model_{args.model_id}' / args.exp_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(checkpoint_dir),
            filename='{epoch:02d}-{val/std_rmse:.4f}',
            monitor='val/std_rmse',
            mode='min',
            save_top_k=3,
            save_last=True,
        ),
        EarlyStopping(
            monitor='val/std_rmse',
            patience=args.patience,
            mode='min',
            verbose=True,
        ),
        LearningRateMonitor(logging_interval='epoch'),
    ]

    # Logger
    logger = TensorBoardLogger(
        save_dir='logs',
        name=f'model_{args.model_id}',
        version=args.exp_name,
    )

    # PyTorch Lightning Trainer
    print("\n[4/4] Starting training...")
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        callbacks=callbacks,
        logger=logger,
        precision='32',  # Use 32-bit for stability
        gradient_clip_val=1.0,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )

    # Train
    trainer.fit(trainer_module, data_module)

    # Save best checkpoint path
    best_ckpt = checkpoint_dir / 'best.ckpt'
    if trainer.checkpoint_callback.best_model_path:
        import shutil
        shutil.copy(trainer.checkpoint_callback.best_model_path, best_ckpt)
        print(f"\nBest checkpoint saved to: {best_ckpt}")

    # Print results
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best checkpoint: {checkpoint_dir}")
    print(f"Best val/std_rmse: {trainer.callback_metrics.get('val/std_rmse', 'N/A')}")
    print("=" * 60)


if __name__ == '__main__':
    main()
