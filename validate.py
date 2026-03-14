"""
Validation script to get combined std_rmse score.
This helps compare validation score to competition score for overfitting detection.
"""

import os
import argparse
import torch
import pytorch_lightning as pl
from src.data.graph_builder import FloodGraphBuilder
from src.data.dataset import FloodDataModule
from src.training.trainer import FloodTrainer


def main():
    parser = argparse.ArgumentParser(description="Validate model on held-out data")

    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--model_id", type=int, required=True, help="Model ID (1 or 2)")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")

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
    print(f"Loading checkpoint: {args.checkpoint}")

    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    hparams = checkpoint['hyper_parameters']

    model = FloodTrainer(**hparams)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model.to(device)

    print(f"\nHyperparameters:")
    print(f"  prefix_len: {hparams.get('prefix_len', 8)}")
    print(f"  rollout_steps: {hparams.get('rollout_steps', 8)}")
    print(f"  seq_len: {hparams.get('seq_len', 16)}")
    print(f"  std_1d: {hparams.get('std_1d', 1.0)}")
    print(f"  std_2d: {hparams.get('std_2d', 1.0)}")

    # Build graph
    print(f"\nBuilding graph for Model {args.model_id}...")
    builder = FloodGraphBuilder(args.data_dir, args.model_id)
    graph = builder.build(split="train")
    model.set_graph(graph)

    # Setup data module
    print("Setting up data module...")
    dm = FloodDataModule(
        data_dir=args.data_dir,
        model_id=args.model_id,
        batch_size=args.batch_size,
        seq_len=hparams.get('seq_len', 16),
        pred_len=hparams.get('pred_len', 1),
        stride=4,
        val_ratio=0.2,
        num_workers=0,
    )
    dm.setup("fit")

    # Run validation
    print("\nRunning validation...")
    trainer = pl.Trainer(
        accelerator="mps" if str(device) == "mps" else "auto",
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )

    results = trainer.validate(model, dm.val_dataloader())

    print("\n" + "="*50)
    print("VALIDATION RESULTS")
    print("="*50)
    for key, value in results[0].items():
        print(f"  {key}: {value:.6f}")

    print("\n" + "="*50)
    print("REFERENCE FOR OVERFITTING DETECTION")
    print("="*50)
    print(f"If val/std_rmse >> competition score, model is underfitting")
    print(f"If val/std_rmse << competition score, model is overfitting")
    print(f"Ideal: val/std_rmse ≈ competition score")


if __name__ == "__main__":
    main()
