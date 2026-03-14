#!/usr/bin/env python3
"""
Training script for Graph-TFT model.

Usage:
    python train_graph_tft.py --model_id 1 --exp_name graph_tft_v1 --max_epochs 30
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
from src.models.graph_tft import GraphTFT
from src.training.graph_tft_trainer import GraphTFTTrainer


def parse_args():
    parser = argparse.ArgumentParser(description='Train Graph-TFT model')

    # Data
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Path to data directory')
    parser.add_argument('--model_id', type=int, required=True, choices=[1, 2],
                        help='Model ID (1 or 2)')

    # Model architecture
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension')
    parser.add_argument('--event_latent_dim', type=int, default=16,
                        help='Event latent dimension')
    parser.add_argument('--num_gnn_layers', type=int, default=3,
                        help='Number of GNN layers')
    parser.add_argument('--num_tft_layers', type=int, default=2,
                        help='Number of TFT LSTM layers')
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
    parser.add_argument('--beta', type=float, default=0.01,
                        help='KL loss weight')
    parser.add_argument('--horizon_weighting', type=str, default='linear',
                        choices=['uniform', 'linear', 'exp'],
                        help='Horizon weighting strategy')
    parser.add_argument('--loss_type', type=str, default='mse',
                        choices=['mse', 'huber', 'weighted_mse'],
                        help='Loss function type')
    parser.add_argument('--huber_delta', type=float, default=1.0,
                        help='Delta for Huber loss')
    parser.add_argument('--high_water_weight', type=float, default=2.0,
                        help='Extra weight for high water levels in weighted_mse')

    # Data loading
    parser.add_argument('--seq_len', type=int, default=10,
                        help='Input sequence length (prefix)')
    parser.add_argument('--stride', type=int, default=4,
                        help='Stride for data sampling')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')

    # Experiment
    parser.add_argument('--exp_name', type=str, default='graph_tft',
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
            'hidden_dim': max(args.hidden_dim, 96),  # Increase capacity
            'num_gnn_layers': max(args.num_gnn_layers, 4),  # More spatial propagation
            'num_tft_layers': max(args.num_tft_layers, 3),  # More temporal capacity
            'dropout': min(args.dropout, 0.15),  # Less dropout for larger model
        }
    else:
        # Model 1: Default settings
        return {
            'hidden_dim': args.hidden_dim,
            'num_gnn_layers': args.num_gnn_layers,
            'num_tft_layers': args.num_tft_layers,
            'dropout': args.dropout,
        }


def main():
    args = parse_args()

    # Set random seed
    pl.seed_everything(42, workers=True)

    # Get model-specific config
    model_config = get_model_specific_config(args.model_id, args)

    print(f"=" * 60)
    print(f"Training Graph-TFT for Model {args.model_id}")
    print(f"Experiment: {args.exp_name}")
    print(f"Model-specific config: hidden_dim={model_config['hidden_dim']}, "
          f"gnn_layers={model_config['num_gnn_layers']}, tft_layers={model_config['num_tft_layers']}")
    print(f"=" * 60)

    # Check CUDA availability
    if torch.cuda.is_available():
        print(f"CUDA available: {torch.cuda.get_device_name(0)}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif torch.backends.mps.is_available():
        print("MPS (Apple Silicon) available")
    else:
        print("Running on CPU")

    # Create data module (builds graph internally)
    print("\n[2/4] Loading data...")
    data_module = FloodDataModule(
        data_dir=args.data_dir,
        model_id=args.model_id,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        pred_len=args.prediction_horizon,  # Multi-horizon prediction
        stride=args.stride,
        num_workers=args.num_workers,
        graph_kwargs={'add_knn_2d_edges': True, 'knn_k': 8},
    )
    data_module.setup('fit')

    # Get normalization stats and graph from data module
    norm_stats = data_module.norm_stats
    graph = data_module.graph
    print(f"  Training samples: {sum(len(ds) for ds in data_module.train_datasets)}")
    print(f"  Validation samples: {sum(len(ds) for ds in data_module.val_datasets)}")

    # Get feature dimensions from graph
    print(f"\n[2/4] Building heterogeneous graph...")
    static_1d_dim = graph['1d'].x.shape[1]
    static_2d_dim = graph['2d'].x.shape[1]
    print(f"  1D nodes: {graph['1d'].x.shape[0]}, features: {static_1d_dim}")
    print(f"  2D nodes: {graph['2d'].x.shape[0]}, features: {static_2d_dim}")

    # Dynamic feature dimensions (from dataset)
    # 1D: [water_level, inlet_flow] = 2
    # 2D: [rainfall, water_level, water_volume] = 3
    dynamic_1d_dim = 2
    dynamic_2d_dim = 3

    # Move graph to device
    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    graph = graph.to(device)

    # Create model with model-specific hyperparameters
    print("\n[3/4] Creating Graph-TFT model...")
    model = GraphTFT(
        static_1d_dim=static_1d_dim,
        static_2d_dim=static_2d_dim,
        dynamic_1d_dim=dynamic_1d_dim,
        dynamic_2d_dim=dynamic_2d_dim,
        hidden_dim=model_config['hidden_dim'],
        event_latent_dim=args.event_latent_dim,
        num_gnn_layers=model_config['num_gnn_layers'],
        num_tft_layers=model_config['num_tft_layers'],
        num_heads=args.num_heads,
        prediction_horizon=args.prediction_horizon,
        use_attention=True,
        use_event_latent=True,
        dropout=model_config['dropout'],
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total parameters: {num_params:,}")
    print(f"  Trainable parameters: {num_trainable:,}")

    # Create trainer module
    trainer_module = GraphTFTTrainer(
        model=model,
        graph=graph,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        beta=args.beta,
        horizon_weighting=args.horizon_weighting,
        max_epochs=args.max_epochs,
        norm_stats=norm_stats,
        loss_type=args.loss_type,
        huber_delta=args.huber_delta,
        high_water_weight=args.high_water_weight,
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

    # Print results
    print("\n" + "=" * 60)
    print("Training completed!")
    print(f"Best checkpoint: {checkpoint_dir}")
    print(f"Best val/std_rmse: {trainer.callback_metrics.get('val/std_rmse', 'N/A')}")
    print("=" * 60)


if __name__ == '__main__':
    main()
