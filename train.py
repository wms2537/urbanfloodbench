"""
Main training script for CL-DTS Urban Flood Model.
"""

import os
import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from src.data.graph_builder import FloodGraphBuilder
from src.data.dataset import FloodDataModule
from src.training.trainer import FloodTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train CL-DTS Flood Model")

    # Data
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--model_id", type=int, default=1, choices=[1, 2], help="Model ID (1 or 2)")

    # Model architecture
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--latent_dim", type=int, default=32, help="Latent dimension")
    parser.add_argument("--event_latent_dim", type=int, default=16, help="Event latent dimension")
    parser.add_argument("--num_gnn_layers", type=int, default=3, help="Number of GNN layers")
    parser.add_argument("--num_temporal_layers", type=int, default=2, help="Number of temporal layers")
    parser.add_argument("--use_attention", action="store_true", default=True, help="Use attention in GNN")
    parser.add_argument("--use_event_latent", action="store_true", default=True, help="Use event latent")
    parser.add_argument("--use_dynamic_latent", action="store_true", default=False, help="Use dynamic latent")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate (increased for regularization)")

    # Training
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size (reduced for longer sequences)")
    parser.add_argument("--seq_len", type=int, default=48, help="Input sequence length (increased for longer rollout)")
    parser.add_argument("--pred_len", type=int, default=1, help="Prediction length")
    parser.add_argument("--prefix_len", type=int, default=10, help="Prefix for event encoding (matches test warmup)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay (increased for regularization)")
    parser.add_argument("--beta", type=float, default=0.1, help="KL weight for ELBO")
    parser.add_argument("--rollout_steps", type=int, default=32, help="Rollout steps for training (closer to test length)")
    parser.add_argument("--max_epochs", type=int, default=50, help="Maximum epochs")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")

    # Scheduled sampling (teacher forcing decay)
    parser.add_argument("--initial_tf", type=float, default=1.0, help="Initial teacher forcing ratio")
    parser.add_argument("--min_tf", type=float, default=0.0, help="Minimum teacher forcing ratio")
    parser.add_argument("--tf_decay_epochs", type=int, default=30, help="Epochs to decay teacher forcing")

    # Graph construction
    parser.add_argument("--coupling_k", type=int, default=3, help="kNN for 1D-2D coupling")
    parser.add_argument("--add_knn_2d", action="store_true", help="Add kNN edges to 2D")
    parser.add_argument("--knn_k", type=int, default=8, help="kNN k for 2D edges")

    # System
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--accelerator", type=str, default="auto", help="Accelerator")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices")
    parser.add_argument("--precision", type=str, default="32", help="Precision (32, 16-mixed, bf16-mixed)")
    parser.add_argument("--exp_name", type=str, default="cldts_v1", help="Experiment name")
    parser.add_argument("--stride", type=int, default=None, help="Stride for sliding window (default: pred_len)")
    parser.add_argument("--gradient_clip", type=float, default=1.0, help="Gradient clipping value")
    parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Accumulate gradients over N batches")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set seed
    pl.seed_everything(42)

    print(f"Training CL-DTS for Model {args.model_id}")
    print(f"Data directory: {args.data_dir}")

    # Standard deviation values for standardized RMSE (competition metric)
    # Computed from training data water levels per (model, node_type)
    STD_VALUES = {
        1: {'1d': 16.877747, '2d': 14.378797},
        2: {'1d': 3.191784, '2d': 2.727131},
    }
    std_1d = STD_VALUES[args.model_id]['1d']
    std_2d = STD_VALUES[args.model_id]['2d']
    print(f"Standardization: std_1d={std_1d:.4f}, std_2d={std_2d:.4f}")

    # Build graph
    print("Building graph...")
    graph_builder = FloodGraphBuilder(
        data_dir=args.data_dir,
        model_id=args.model_id,
        coupling_k=args.coupling_k,
        add_knn_2d_edges=args.add_knn_2d,
        knn_k=args.knn_k,
    )
    graph = graph_builder.build(split="train")
    print(f"Graph built: {graph}")

    # Create data module
    print("Creating data module...")
    data_module = FloodDataModule(
        data_dir=args.data_dir,
        model_id=args.model_id,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        stride=args.stride if args.stride else args.pred_len,  # Configurable stride
        num_workers=args.num_workers,
    )
    data_module.setup()

    # Get feature dimensions from data
    # Use a sample to determine dimensions
    sample_ds = data_module.train_datasets[0]
    sample = sample_ds[0]

    static_1d_dim = graph['1d'].x.shape[1]
    static_2d_dim = graph['2d'].x.shape[1]
    dynamic_1d_dim = sample['input_1d'].shape[-1]
    dynamic_2d_dim = sample['input_2d'].shape[-1]

    print(f"Feature dimensions:")
    print(f"  Static 1D: {static_1d_dim}, Static 2D: {static_2d_dim}")
    print(f"  Dynamic 1D: {dynamic_1d_dim}, Dynamic 2D: {dynamic_2d_dim}")

    # Create trainer
    model = FloodTrainer(
        static_1d_dim=static_1d_dim,
        static_2d_dim=static_2d_dim,
        dynamic_1d_dim=dynamic_1d_dim,
        dynamic_2d_dim=dynamic_2d_dim,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        event_latent_dim=args.event_latent_dim,
        num_gnn_layers=args.num_gnn_layers,
        num_temporal_layers=args.num_temporal_layers,
        use_attention=args.use_attention,
        use_event_latent=args.use_event_latent,
        use_dynamic_latent=args.use_dynamic_latent,
        dropout=args.dropout,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        beta=args.beta,
        rollout_steps=args.rollout_steps,
        # Scheduled sampling parameters
        initial_teacher_forcing=args.initial_tf,
        min_teacher_forcing=args.min_tf,
        teacher_forcing_decay_epochs=args.tf_decay_epochs,
        seq_len=args.seq_len,
        pred_len=args.pred_len,
        prefix_len=args.prefix_len,
        std_1d=std_1d,
        std_2d=std_2d,
    )

    # Set graph and normalization stats
    model.set_graph(graph)
    model.set_normalization_stats(data_module.norm_stats)

    # Callbacks - monitor standardized RMSE (competition metric)
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"checkpoints/{args.exp_name}/model_{args.model_id}",
        filename="{epoch:02d}-{val/std_rmse:.4f}",
        monitor="val/std_rmse",
        mode="min",
        save_top_k=3,
        save_last=True,
    )

    early_stopping = EarlyStopping(
        monitor="val/std_rmse",
        mode="min",
        patience=args.patience,
        verbose=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Logger
    logger = TensorBoardLogger(
        save_dir="logs",
        name=args.exp_name,
        version=f"model_{args.model_id}",
    )

    # PyTorch Lightning Trainer
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback, early_stopping, lr_monitor],
        logger=logger,
        gradient_clip_val=args.gradient_clip,
        accumulate_grad_batches=args.accumulate_grad_batches,
        log_every_n_steps=10,
        enable_progress_bar=True,
    )

    # Train
    print("Starting training...")
    trainer.fit(model, data_module)

    # Test
    print("Running test...")
    trainer.test(model, data_module)

    print(f"Training complete. Best checkpoint: {checkpoint_callback.best_model_path}")


if __name__ == "__main__":
    main()
