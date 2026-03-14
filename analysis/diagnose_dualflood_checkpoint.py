#!/usr/bin/env python3
"""
Diagnose DualFlood checkpoint behavior on validation events.

Focus:
- Overall standardized RMSE (same normalized metric family as training)
- Per-event breakdown by target horizon length
- Per-lead-time RMSE profile to expose rollout drift
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List
import sys

import numpy as np
import pandas as pd
import torch

# Allow running as: python analysis/diagnose_dualflood_checkpoint.py
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from predict_dual_flood import (
    load_architecture_flags_from_checkpoint,
    load_model_state_dict,
    load_norm_stats_from_checkpoint,
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


def _horizon_bucket(horizon: int) -> str:
    if horizon <= 100:
        return "short(<=100)"
    if horizon <= 220:
        return "mid(101-220)"
    return "long(>220)"


def build_model_and_data(args):
    device = torch.device(args.device) if args.device else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    norm_stats = load_norm_stats_from_checkpoint(args.checkpoint)
    static_norm = load_static_norm_stats_from_checkpoint(args.checkpoint)
    if static_norm is None:
        tmp_builder = DualFloodGraphBuilder(args.data_dir, args.model_id)
        static_norm = tmp_builder.compute_static_norm_stats()

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
        use_norm_cache=True,
        future_inlet_mode_train=args.future_inlet_mode_train,
        future_inlet_mode_val=args.future_inlet_mode_val,
        train_start_only=args.train_start_only,
        val_start_only=args.val_start_only,
    )
    dm.setup()

    graph = DualFloodGraphBuilder(args.data_dir, args.model_id, static_norm_stats=static_norm).build(split="train")
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
        transition_scale=args.transition_scale,
        coupling_scale=args.coupling_scale,
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
    _, _ = load_matching_state_dict(model, model_state)
    model = model.to(device)
    model.eval()

    graph = graph.to(device)

    return model, graph, dm, device


def main():
    parser = argparse.ArgumentParser(description="Diagnose DualFlood checkpoint on validation data.")
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--model_id", default=2, type=int)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--pred_len", default=399, type=int)
    parser.add_argument("--transition_scale", default=0.05, type=float)
    parser.add_argument("--coupling_scale", default=0.03, type=float)
    parser.add_argument("--future_inlet_mode_train", default="mixed", type=str)
    parser.add_argument("--future_inlet_mode_val", default="missing", type=str)
    parser.add_argument("--train_start_only", action="store_true")
    parser.add_argument("--val_start_only", action="store_true")
    parser.add_argument("--device", default="", type=str)
    parser.add_argument("--output_csv", default="", type=str)
    args = parser.parse_args()

    model, graph, dm, device = build_model_and_data(args)

    tmax = args.pred_len
    sum_sq_1d = np.zeros(tmax, dtype=np.float64)
    cnt_1d = np.zeros(tmax, dtype=np.float64)
    sum_sq_2d = np.zeros(tmax, dtype=np.float64)
    cnt_2d = np.zeros(tmax, dtype=np.float64)

    rows: List[Dict] = []

    with torch.no_grad():
        for ds in dm.val_datasets:
            for idx in range(len(ds)):
                sample = ds[idx]
                batch = _to_device_batch(sample, device)
                horizon = int(batch["target_len"].item())
                outputs = model(
                    graph=graph,
                    input_1d=batch["input_1d"],
                    input_2d=batch["input_2d"],
                    future_rainfall=batch["future_rainfall"],
                    future_inlet=batch["future_inlet"],
                    future_inlet_mask=batch["future_inlet_mask"],
                    rollout_len=horizon,
                )

                pred_1d = outputs["pred_wl_1d"][0, :horizon, :, 0]
                pred_2d = outputs["pred_wl_2d"][0, :horizon, :, 0]
                tgt_1d = batch["target_wl_1d"][0, :horizon, :, 0]
                tgt_2d = batch["target_wl_2d"][0, :horizon, :, 0]

                err_1d = pred_1d - tgt_1d
                err_2d = pred_2d - tgt_2d

                mse_1d = err_1d.pow(2).mean()
                mse_2d = err_2d.pow(2).mean()
                rmse_1d = torch.sqrt(mse_1d.clamp_min(1e-12)).item()
                rmse_2d = torch.sqrt(mse_2d.clamp_min(1e-12)).item()
                std_rmse = 0.5 * (rmse_1d + rmse_2d)

                rows.append(
                    {
                        "event_id": int(ds.event_id),
                        "sample_idx": int(idx),
                        "horizon": horizon,
                        "bucket": _horizon_bucket(horizon),
                        "rmse_norm_1d": rmse_1d,
                        "rmse_norm_2d": rmse_2d,
                        "std_rmse": std_rmse,
                    }
                )

                err_1d_np = err_1d.detach().cpu().numpy()
                err_2d_np = err_2d.detach().cpu().numpy()
                for t in range(horizon):
                    e1 = err_1d_np[t]
                    e2 = err_2d_np[t]
                    sum_sq_1d[t] += float((e1 * e1).sum())
                    cnt_1d[t] += float(e1.size)
                    sum_sq_2d[t] += float((e2 * e2).sum())
                    cnt_2d[t] += float(e2.size)

    df = pd.DataFrame(rows)
    df = df.sort_values(["event_id", "sample_idx"]).reset_index(drop=True)

    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Samples evaluated: {len(df)}")
    print(f"Overall std_rmse (mean): {df['std_rmse'].mean():.6f}")
    print("=" * 70)

    print("\nBy horizon bucket:")
    print(
        df.groupby("bucket", as_index=False)["std_rmse"]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
        .to_string(index=False)
    )

    lead_steps = [1, 10, 30, 60, 90, 120, 180, 240, 300, 360, 399]
    print("\nLead-time profile (normalized RMSE):")
    for step in lead_steps:
        t = step - 1
        if t < 0 or t >= tmax:
            continue
        if cnt_1d[t] <= 0 or cnt_2d[t] <= 0:
            continue
        rmse_t_1d = np.sqrt(sum_sq_1d[t] / max(cnt_1d[t], 1.0))
        rmse_t_2d = np.sqrt(sum_sq_2d[t] / max(cnt_2d[t], 1.0))
        std_rmse_t = 0.5 * (rmse_t_1d + rmse_t_2d)
        print(f"  t={step:>3}: std_rmse={std_rmse_t:.6f} (1d={rmse_t_1d:.6f}, 2d={rmse_t_2d:.6f})")

    if args.output_csv:
        out_path = Path(args.output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"\nSaved per-sample diagnostics: {out_path}")


if __name__ == "__main__":
    main()
