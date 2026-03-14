#!/usr/bin/env python3
"""
Analyze DualFlood MoE routing behavior on validation samples.

Reports:
- top-1 expert usage frequencies for each transition block
- average routing entropy
- fraction of calls with dominant expert probability > threshold
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, List
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from predict_dual_flood import (
    load_architecture_flags_from_checkpoint,
    load_model_state_dict,
    load_static_norm_stats_from_checkpoint,
)
from train_dual_flood import (
    DualFloodDataModule,
    DualFloodGNN,
    DualFloodGraphBuilder,
    load_matching_state_dict,
)


def _to_device_batch(sample: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for key, value in sample.items():
        if torch.is_tensor(value):
            if value.ndim == 0:
                out[key] = value.to(device=device)
            else:
                out[key] = value.unsqueeze(0).to(device=device)
        else:
            out[key] = value
    return out


def main():
    parser = argparse.ArgumentParser(description="Analyze MoE routing usage for DualFlood checkpoint.")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--model_id", default=2, type=int)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--pred_len", default=399, type=int)
    parser.add_argument("--max_samples", default=0, type=int, help="0 = all validation samples")
    parser.add_argument("--future_inlet_mode_train", default="mixed", type=str)
    parser.add_argument("--future_inlet_mode_val", default="missing", type=str)
    parser.add_argument("--train_start_only", action="store_true")
    parser.add_argument("--val_start_only", action="store_true")
    parser.add_argument("--device", default="", type=str)
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    static_norm = load_static_norm_stats_from_checkpoint(args.checkpoint)
    if static_norm is None:
        static_norm = DualFloodGraphBuilder(args.data_dir, args.model_id).compute_static_norm_stats()

    dm = DualFloodDataModule(
        data_dir=args.data_dir,
        model_id=args.model_id,
        batch_size=1,
        seq_len=10,
        pred_len=args.pred_len,
        min_pred_len=1,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False,
        future_inlet_mode_train=args.future_inlet_mode_train,
        future_inlet_mode_val=args.future_inlet_mode_val,
        train_start_only=args.train_start_only,
        val_start_only=args.val_start_only,
    )
    dm.setup()
    graph = DualFloodGraphBuilder(args.data_dir, args.model_id, static_norm_stats=static_norm).build(split="train")
    graph = graph.to(device)

    edge_dim_1d = graph["1d", "pipe", "1d"].edge_attr.shape[1]
    edge_dim_2d = graph["2d", "surface", "2d"].edge_attr.shape[1]
    arch = load_architecture_flags_from_checkpoint(args.checkpoint)
    hidden_dim = 96 if args.model_id == 2 else 64
    latent_dim = 48 if args.model_id == 2 else 32
    num_gnn_layers = 4 if args.model_id == 2 else 3

    model = DualFloodGNN(
        num_1d_nodes=dm.num_1d_nodes,
        num_2d_nodes=dm.num_2d_nodes,
        num_1d_edges=dm.num_1d_edges,
        num_2d_edges=dm.num_2d_edges,
        edge_dim_1d=edge_dim_1d,
        edge_dim_2d=edge_dim_2d,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_gnn_layers=num_gnn_layers,
        pred_len=args.pred_len,
        transition_scale=0.05,
        coupling_scale=0.03,
        use_flow_first_decoder=arch["use_flow_first_decoder"],
        use_multiscale_2d=arch["use_multiscale_2d"],
        multiscale_num_clusters=arch["multiscale_num_clusters"],
        use_moe_transition=arch["use_moe_transition"],
        moe_num_experts=arch["moe_num_experts"],
        moe_mode=arch["moe_mode"],
        moe_top_k=arch["moe_top_k"],
        use_dual_timescale_latent=arch["use_dual_timescale_latent"],
        slow_timescale_ratio=arch["slow_timescale_ratio"],
        use_direct_ar_hybrid=arch["use_direct_ar_hybrid"],
        use_inlet_imputer=arch["use_inlet_imputer"],
        direct_ar_init_blend=arch["direct_ar_init_blend"],
        precompute_transition_controls=arch["precompute_transition_controls"],
        precompute_decoder_edge_terms=arch["precompute_decoder_edge_terms"],
        use_stable_flow_rollout=arch["use_stable_flow_rollout"],
    )
    model_state = load_model_state_dict(args.checkpoint)
    load_matching_state_dict(model, model_state)
    model = model.to(device)
    model.eval()

    if not arch["use_moe_transition"]:
        print("Checkpoint/model does not use MoE transitions. Nothing to analyze.")
        return

    collectors = defaultdict(list)

    def make_hook(name: str):
        def _hook(module, inputs, output):
            collectors[name].append(output.detach().cpu())
        return _hook

    hooks = []
    hooks.append(model.transition_1d.regime_gate.register_forward_hook(make_hook("fast_1d")))
    hooks.append(model.transition_2d.regime_gate.register_forward_hook(make_hook("fast_2d")))
    if arch["use_dual_timescale_latent"]:
        hooks.append(model.transition_1d_slow.regime_gate.register_forward_hook(make_hook("slow_1d")))
        hooks.append(model.transition_2d_slow.regime_gate.register_forward_hook(make_hook("slow_2d")))

    processed = 0
    with torch.no_grad():
        for ds in dm.val_datasets:
            for idx in range(len(ds)):
                sample = ds[idx]
                batch = _to_device_batch(sample, device)
                horizon = int(batch["target_len"].item())
                _ = model(
                    graph=graph,
                    input_1d=batch["input_1d"],
                    input_2d=batch["input_2d"],
                    future_rainfall=batch["future_rainfall"],
                    future_inlet=batch["future_inlet"],
                    future_inlet_mask=batch["future_inlet_mask"],
                    rollout_len=horizon,
                )
                processed += 1
                if args.max_samples > 0 and processed >= args.max_samples:
                    break
            if args.max_samples > 0 and processed >= args.max_samples:
                break

    for h in hooks:
        h.remove()

    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Processed validation samples: {processed}")
    print("=" * 70)

    for name, lst in collectors.items():
        if not lst:
            print(f"\n{name}: no routing calls captured")
            continue
        logits = torch.cat(lst, dim=0)  # [num_calls, num_experts]
        probs = torch.softmax(logits, dim=-1)
        top1 = torch.argmax(probs, dim=-1)
        num_experts = probs.shape[1]
        counts = torch.bincount(top1, minlength=num_experts).float()
        freq = counts / counts.sum().clamp_min(1.0)
        entropy = -(probs * (probs.clamp_min(1e-8).log())).sum(dim=-1)
        max_prob = probs.max(dim=-1).values
        dom80 = (max_prob > 0.80).float().mean().item()
        dom90 = (max_prob > 0.90).float().mean().item()

        print(f"\n{name}:")
        print(f"  calls: {probs.shape[0]}")
        print(f"  mean_entropy: {entropy.mean().item():.6f}")
        print(f"  dominant_prob>0.80: {dom80:.3f}")
        print(f"  dominant_prob>0.90: {dom90:.3f}")
        print("  top1_expert_frequency:")
        for i in range(num_experts):
            print(f"    expert_{i}: {freq[i].item():.4f}")


if __name__ == "__main__":
    main()
