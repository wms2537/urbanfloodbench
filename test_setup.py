"""
Test script to verify data loading and model forward pass.
"""

import os
import sys
import torch

print("Testing CL-DTS setup...")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"MPS available: {torch.backends.mps.is_available()}")

# Test imports
print("\n1. Testing imports...")
try:
    from src.data.graph_builder import FloodGraphBuilder
    from src.data.dataset import FloodEventDataset, FloodDataModule
    from src.models.coupled_gnn import CoupledHeteroGNN
    from src.models.temporal import TemporalBlock, SpatioTemporalEncoder
    from src.models.cldts import CLDTS
    from src.training.trainer import FloodTrainer
    from src.training.losses import CombinedLoss
    print("   All imports successful!")
except ImportError as e:
    print(f"   Import error: {e}")
    sys.exit(1)

# Test graph building
print("\n2. Testing graph construction...")
DATA_DIR = "data"

try:
    for model_id in [1, 2]:
        builder = FloodGraphBuilder(DATA_DIR, model_id)
        graph = builder.build(split="train")
        print(f"   Model {model_id}:")
        print(f"     1D nodes: {graph['1d'].num_nodes}, features: {graph['1d'].x.shape}")
        print(f"     2D nodes: {graph['2d'].num_nodes}, features: {graph['2d'].x.shape}")
        print(f"     1D edges: {graph['1d', 'pipe', '1d'].edge_index.shape[1]}")
        print(f"     2D edges: {graph['2d', 'surface', '2d'].edge_index.shape[1]}")
        print(f"     Coupling edges: {graph['1d', 'couples_to', '2d'].edge_index.shape[1]}")
except Exception as e:
    print(f"   Graph building error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test dataset
print("\n3. Testing dataset loading...")
try:
    # Find first training event
    train_path = os.path.join(DATA_DIR, "Model_1", "train")
    events = [int(d.split("_")[1]) for d in os.listdir(train_path) if d.startswith("event_")]
    event_id = sorted(events)[0]
    print(f"   Loading event {event_id}...")

    builder = FloodGraphBuilder(DATA_DIR, 1)
    graph = builder.build(split="train")

    dataset = FloodEventDataset(
        DATA_DIR, model_id=1, event_id=event_id, split="train",
        graph=graph, seq_len=16, pred_len=1, stride=4, normalize=True
    )

    print(f"   Dataset length: {len(dataset)}")
    sample = dataset[0]
    print(f"   Sample input_1d shape: {sample['input_1d'].shape}")
    print(f"   Sample input_2d shape: {sample['input_2d'].shape}")
    print(f"   Sample target_1d shape: {sample['target_1d'].shape}")
    print(f"   Sample target_2d shape: {sample['target_2d'].shape}")
except Exception as e:
    print(f"   Dataset error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test model forward pass
print("\n4. Testing model forward pass...")
try:
    # Create model
    static_1d_dim = graph['1d'].x.shape[1]
    static_2d_dim = graph['2d'].x.shape[1]
    dynamic_1d_dim = sample['input_1d'].shape[-1]
    dynamic_2d_dim = sample['input_2d'].shape[-1]

    model = CLDTS(
        static_1d_dim=static_1d_dim,
        static_2d_dim=static_2d_dim,
        dynamic_1d_dim=dynamic_1d_dim,
        dynamic_2d_dim=dynamic_2d_dim,
        hidden_dim=32,  # Smaller for testing
        latent_dim=16,
        event_latent_dim=8,
        num_gnn_layers=2,
        num_temporal_layers=1,
        use_attention=True,
        use_event_latent=True,
        use_dynamic_latent=False,
        dropout=0.1,
    )

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")

    # Forward pass
    input_1d = sample['input_1d'].unsqueeze(0)  # Add batch dim
    input_2d = sample['input_2d'].unsqueeze(0)

    with torch.no_grad():
        outputs = model(graph, input_1d, input_2d, prefix_len=8)

    print(f"   Output pred_1d shape: {outputs['pred_1d'].shape}")
    print(f"   Output pred_2d shape: {outputs['pred_2d'].shape}")
    if outputs['c_e'] is not None:
        print(f"   Event latent c_e shape: {outputs['c_e'].shape}")
    print("   Forward pass successful!")
except Exception as e:
    print(f"   Model error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test rollout
print("\n5. Testing autoregressive rollout...")
try:
    with torch.no_grad():
        rollout_outputs = model.rollout(
            graph, input_1d, input_2d, horizon=10, prefix_len=8
        )
    print(f"   Rollout pred_1d shape: {rollout_outputs['pred_1d'].shape}")
    print(f"   Rollout pred_2d shape: {rollout_outputs['pred_2d'].shape}")
    print("   Rollout successful!")
except Exception as e:
    print(f"   Rollout error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test loss computation
print("\n6. Testing loss computation...")
try:
    loss_fn = CombinedLoss(beta=0.1, rollout_steps=4)

    pred_1d = outputs['pred_1d'][:, -1]  # [batch, num_1d_nodes]
    pred_2d = outputs['pred_2d'][:, -1]  # [batch, num_2d_nodes]
    # Squeeze extra dimensions from target and match shape
    target_1d = sample['target_1d'].squeeze(-1).squeeze(0).unsqueeze(0)  # [batch, num_1d_nodes]
    target_2d = sample['target_2d'].squeeze(-1).squeeze(0).unsqueeze(0)  # [batch, num_2d_nodes]

    print(f"   pred_1d shape: {pred_1d.shape}, target_1d shape: {target_1d.shape}")
    print(f"   pred_2d shape: {pred_2d.shape}, target_2d shape: {target_2d.shape}")

    loss, components = loss_fn(
        pred_1d, pred_2d, target_1d, target_2d,
        outputs.get('c_e_mean'), outputs.get('c_e_logvar')
    )
    print(f"   Total loss: {loss.item():.4f}")
    for name, value in components.items():
        print(f"   {name}: {value.item():.4f}")
    print("   Loss computation successful!")
except Exception as e:
    print(f"   Loss error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*50)
print("All tests passed! Ready for training.")
print("="*50)
print("\nTo start training, run:")
print("  python train.py --model_id 1 --max_epochs 10")
print("\nOr for a quick test:")
print("  python train.py --model_id 1 --max_epochs 2 --batch_size 2")
