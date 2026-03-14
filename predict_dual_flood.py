#!/usr/bin/env python3
"""
Generate Model 2 DualFlood predictions and build a competition-ready submission.

Workflow:
1) Run DualFlood inference on Model 2 test events (prefix=10, long rollout)
2) Map predictions to sample_submission row_id for exact ordering
3) Replace Model 2 rows in a base full submission (keeps Model 1 untouched)
4) Save final parquet in official schema
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from train_dual_flood import (
    DualFloodGNN,
    DualFloodGraphBuilder,
    load_matching_state_dict,
    to_numpy_float32,
)


def to_numpy(x) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def load_norm_stats_from_checkpoint(ckpt_path: str) -> Dict:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", {})
    norm_stats = hp.get("norm_stats")
    if norm_stats is None:
        raise ValueError(f"norm_stats not found in checkpoint: {ckpt_path}")

    # Ensure array/scalar types are consistent for vectorized math.
    norm_stats["node_1d"]["mean"] = to_numpy(norm_stats["node_1d"]["mean"])
    norm_stats["node_1d"]["std"] = to_numpy(norm_stats["node_1d"]["std"])
    norm_stats["node_2d"]["mean"] = to_numpy(norm_stats["node_2d"]["mean"])
    norm_stats["node_2d"]["std"] = to_numpy(norm_stats["node_2d"]["std"])
    norm_stats["water_level_1d"]["mean"] = float(norm_stats["water_level_1d"]["mean"])
    norm_stats["water_level_1d"]["std"] = float(norm_stats["water_level_1d"]["std"])
    norm_stats["water_level_2d"]["mean"] = float(norm_stats["water_level_2d"]["mean"])
    norm_stats["water_level_2d"]["std"] = float(norm_stats["water_level_2d"]["std"])
    return norm_stats


def load_static_norm_stats_from_checkpoint(ckpt_path: str) -> Optional[Dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", {})
    stats = hp.get("static_norm_stats")
    if stats is None:
        return None
    out = {
        "node_1d": {
            "mean": to_numpy_float32(stats["node_1d"]["mean"]),
            "std": to_numpy_float32(stats["node_1d"]["std"]),
        },
        "node_2d": {
            "mean": to_numpy_float32(stats["node_2d"]["mean"]),
            "std": to_numpy_float32(stats["node_2d"]["std"]),
        },
    }
    if "edge_1d" in stats:
        out["edge_1d"] = {
            "cols": list(stats["edge_1d"].get("cols", [])),
            "mean": to_numpy_float32(stats["edge_1d"]["mean"]),
            "std": to_numpy_float32(stats["edge_1d"]["std"]),
        }
    if "edge_2d" in stats:
        out["edge_2d"] = {
            "cols": list(stats["edge_2d"].get("cols", [])),
            "mean": to_numpy_float32(stats["edge_2d"]["mean"]),
            "std": to_numpy_float32(stats["edge_2d"]["std"]),
        }
    return out


def load_model_state_dict(ckpt_path: str) -> Dict[str, torch.Tensor]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    raw_state = ckpt["state_dict"]
    model_state = {}
    for key, value in raw_state.items():
        if key.startswith("model."):
            model_state[key[len("model."):]] = value
    if not model_state:
        raise ValueError(f"No model.* weights found in checkpoint: {ckpt_path}")
    return model_state


def load_architecture_flags_from_checkpoint(ckpt_path: str) -> Dict[str, object]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hp = ckpt.get("hyper_parameters", {})
    raw_flags = hp.get("architecture_flags", {})
    if not isinstance(raw_flags, dict):
        raw_flags = {}

    return {
        "use_flow_first_decoder": bool(raw_flags.get("use_flow_first_decoder", False)),
        "use_multiscale_2d": bool(raw_flags.get("use_multiscale_2d", False)),
        "multiscale_num_clusters": int(raw_flags.get("multiscale_num_clusters", 128)),
        "use_moe_transition": bool(raw_flags.get("use_moe_transition", False)),
        "moe_num_experts": int(raw_flags.get("moe_num_experts", 4)),
        "moe_mode": str(raw_flags.get("moe_mode", "dense")),
        "moe_top_k": int(raw_flags.get("moe_top_k", 1)),
        "use_dual_timescale_latent": bool(raw_flags.get("use_dual_timescale_latent", False)),
        "slow_timescale_ratio": float(raw_flags.get("slow_timescale_ratio", 0.25)),
        "use_direct_ar_hybrid": bool(raw_flags.get("use_direct_ar_hybrid", False)),
        "use_inlet_imputer": bool(raw_flags.get("use_inlet_imputer", False)),
        "use_nodewise_1d_dynamics": bool(raw_flags.get("use_nodewise_1d_dynamics", False)),
        "direct_ar_init_blend": float(raw_flags.get("direct_ar_init_blend", 0.5)),
        "use_stable_flow_rollout": bool(raw_flags.get("use_stable_flow_rollout", True)),
        "precompute_transition_controls": bool(raw_flags.get("precompute_transition_controls", True)),
        "precompute_decoder_edge_terms": bool(raw_flags.get("precompute_decoder_edge_terms", True)),
    }


def list_test_events(data_dir: str, model_id: int) -> List[int]:
    test_dir = Path(data_dir) / f"Model_{model_id}" / "test"
    events = sorted(
        int(d.name.split("_")[1]) for d in test_dir.iterdir()
        if d.is_dir() and d.name.startswith("event_")
    )
    if not events:
        raise ValueError(f"No test events found in {test_dir}")
    return events


def reshape_node_dynamic(event_dir: Path, filename: str, cols: List[str]) -> np.ndarray:
    df = pd.read_csv(event_dir / filename)
    df = df.sort_values(["timestep", "node_idx"])
    num_t = int(df["timestep"].nunique())
    num_n = int(df["node_idx"].nunique())
    arr = df[cols].to_numpy(dtype=np.float32).reshape(num_t, num_n, len(cols))
    return np.nan_to_num(arr, nan=0.0)


def predict_event(
    model: DualFloodGNN,
    graph,
    norm_stats: Dict,
    event_dir: Path,
    device: torch.device,
    prefix_len: int = 10,
    future_inlet_mode: str = "missing",
    clamp_min: float = None,
    clamp_max: float = None,
) -> Tuple[np.ndarray, np.ndarray]:
    node_1d = reshape_node_dynamic(
        event_dir, "1d_nodes_dynamic_all.csv", ["water_level", "inlet_flow"]
    )  # [T, N1, 2]
    node_2d = reshape_node_dynamic(
        event_dir, "2d_nodes_dynamic_all.csv", ["water_level", "rainfall", "water_volume"]
    )  # [T, N2, 3]

    if node_1d.shape[0] != node_2d.shape[0]:
        raise ValueError(f"Timestep mismatch in {event_dir}: {node_1d.shape[0]} vs {node_2d.shape[0]}")

    total_t = node_1d.shape[0]
    rollout_len = total_t - prefix_len
    if rollout_len <= 0:
        raise ValueError(f"Invalid rollout length for {event_dir}: total_t={total_t}, prefix_len={prefix_len}")

    n1_mean = norm_stats["node_1d"]["mean"]
    n1_std = norm_stats["node_1d"]["std"]
    n2_mean = norm_stats["node_2d"]["mean"]
    n2_std = norm_stats["node_2d"]["std"]

    # Prefix inputs use the first 10 observed timesteps.
    input_1d = (node_1d[:prefix_len] - n1_mean.reshape(1, 1, -1)) / n1_std.reshape(1, 1, -1)
    input_2d = (node_2d[:prefix_len] - n2_mean.reshape(1, 1, -1)) / n2_std.reshape(1, 1, -1)

    # Future controls: rainfall is known; inlet_flow is NaN on test future and already filled to 0.
    future_rainfall = node_2d[prefix_len:, :, 1:2]
    future_inlet = node_1d[prefix_len:, :, 1:2]
    future_inlet_mask = np.ones_like(future_inlet, dtype=np.float32)

    if future_inlet_mode == "zero":
        future_inlet = np.zeros_like(future_inlet)
        future_inlet_mask = np.ones_like(future_inlet_mask, dtype=np.float32)
    elif future_inlet_mode == "last":
        last_inlet = node_1d[prefix_len - 1:prefix_len, :, 1:2]
        future_inlet = np.repeat(last_inlet, future_inlet.shape[0], axis=0)
        future_inlet_mask = np.ones_like(future_inlet_mask, dtype=np.float32)
    elif future_inlet_mode == "missing":
        # Mean-imputed in normalized space (0 after normalization).
        future_inlet = np.full_like(future_inlet, n1_mean[1], dtype=np.float32)
        future_inlet_mask = np.zeros_like(future_inlet_mask, dtype=np.float32)
    elif future_inlet_mode == "observed":
        future_inlet_mask = np.ones_like(future_inlet_mask, dtype=np.float32)
    else:
        raise ValueError(
            f"Invalid future_inlet_mode='{future_inlet_mode}'. "
            "Expected one of: observed, missing, zero, last"
        )

    future_rainfall = (future_rainfall - n2_mean[1]) / n2_std[1]
    future_inlet = (future_inlet - n1_mean[1]) / n1_std[1]

    input_1d_t = torch.from_numpy(input_1d).unsqueeze(0).to(device=device, dtype=torch.float32)
    input_2d_t = torch.from_numpy(input_2d).unsqueeze(0).to(device=device, dtype=torch.float32)
    future_r_t = torch.from_numpy(future_rainfall).unsqueeze(0).to(device=device, dtype=torch.float32)
    future_i_t = torch.from_numpy(future_inlet).unsqueeze(0).to(device=device, dtype=torch.float32)
    future_m_t = torch.from_numpy(future_inlet_mask).unsqueeze(0).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        outputs = model(
            graph=graph,
            input_1d=input_1d_t,
            input_2d=input_2d_t,
            future_rainfall=future_r_t,
            future_inlet=future_i_t,
            future_inlet_mask=future_m_t,
            rollout_len=rollout_len,
        )

    wl_1d = outputs["pred_wl_1d"][0, :, :, 0].detach().cpu().numpy().astype(np.float32)
    wl_2d = outputs["pred_wl_2d"][0, :, :, 0].detach().cpu().numpy().astype(np.float32)

    wl_1d = wl_1d * norm_stats["water_level_1d"]["std"] + norm_stats["water_level_1d"]["mean"]
    wl_2d = wl_2d * norm_stats["water_level_2d"]["std"] + norm_stats["water_level_2d"]["mean"]

    if clamp_min is not None:
        wl_1d = np.maximum(wl_1d, clamp_min)
        wl_2d = np.maximum(wl_2d, clamp_min)
    if clamp_max is not None:
        wl_1d = np.minimum(wl_1d, clamp_max)
        wl_2d = np.minimum(wl_2d, clamp_max)

    return wl_1d, wl_2d


def load_sample_event_rows(sample_submission: str, model_id: int, event_id: int) -> pd.DataFrame:
    df = pd.read_parquet(
        sample_submission,
        columns=["row_id", "model_id", "event_id", "node_type", "node_id"],
        filters=[("model_id", "==", model_id), ("event_id", "==", event_id)],
    )
    df = df.sort_values("row_id", kind="stable").reset_index(drop=True)
    df["timestep"] = df.groupby(["node_type", "node_id"]).cumcount().astype(np.int32)
    return df


def map_event_predictions_to_rows(
    sample_event: pd.DataFrame,
    pred_1d: np.ndarray,
    pred_2d: np.ndarray,
) -> pd.DataFrame:
    node_type = sample_event["node_type"].to_numpy(dtype=np.int16)
    node_id = sample_event["node_id"].to_numpy(dtype=np.int32)
    timestep = sample_event["timestep"].to_numpy(dtype=np.int32)

    out = np.empty(len(sample_event), dtype=np.float32)
    mask_1d = node_type == 1
    mask_2d = node_type == 2

    if mask_1d.any():
        out[mask_1d] = pred_1d[timestep[mask_1d], node_id[mask_1d]]
    if mask_2d.any():
        out[mask_2d] = pred_2d[timestep[mask_2d], node_id[mask_2d]]

    return pd.DataFrame({
        "row_id": sample_event["row_id"].to_numpy(dtype=np.int64),
        "water_level": out,
    })


def build_model(args, graph, device: torch.device) -> DualFloodGNN:
    edge_dim_1d = graph["1d", "pipe", "1d"].edge_attr.shape[1]
    edge_dim_2d = graph["2d", "surface", "2d"].edge_attr.shape[1]
    arch_flags = load_architecture_flags_from_checkpoint(args.checkpoint)

    model = DualFloodGNN(
        num_1d_nodes=graph["1d"].num_nodes,
        num_2d_nodes=graph["2d"].num_nodes,
        num_1d_edges=graph["1d", "pipe", "1d"].edge_index.shape[1],
        num_2d_edges=graph["2d", "surface", "2d"].edge_index.shape[1],
        edge_dim_1d=edge_dim_1d,
        edge_dim_2d=edge_dim_2d,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_gnn_layers=args.num_gnn_layers,
        pred_len=args.pred_len,
        transition_scale=args.transition_scale,
        coupling_scale=args.coupling_scale,
        use_flow_first_decoder=arch_flags["use_flow_first_decoder"],
        use_multiscale_2d=arch_flags["use_multiscale_2d"],
        multiscale_num_clusters=arch_flags["multiscale_num_clusters"],
        use_moe_transition=arch_flags["use_moe_transition"],
        moe_num_experts=arch_flags["moe_num_experts"],
        moe_mode=arch_flags["moe_mode"],
        moe_top_k=arch_flags["moe_top_k"],
        use_dual_timescale_latent=arch_flags["use_dual_timescale_latent"],
        slow_timescale_ratio=arch_flags["slow_timescale_ratio"],
        use_direct_ar_hybrid=arch_flags["use_direct_ar_hybrid"],
        use_inlet_imputer=arch_flags["use_inlet_imputer"],
        use_nodewise_1d_dynamics=arch_flags["use_nodewise_1d_dynamics"],
        direct_ar_init_blend=arch_flags["direct_ar_init_blend"],
        precompute_transition_controls=arch_flags["precompute_transition_controls"],
        precompute_decoder_edge_terms=arch_flags["precompute_decoder_edge_terms"],
        use_stable_flow_rollout=arch_flags["use_stable_flow_rollout"],
    )

    print(
        "Architecture from checkpoint: "
        f"flow_first={arch_flags['use_flow_first_decoder']}, "
        f"multiscale2d={arch_flags['use_multiscale_2d']}, "
        f"moe={arch_flags['use_moe_transition']}, "
        f"moe_mode={arch_flags['moe_mode']}, "
        f"moe_top_k={arch_flags['moe_top_k']}, "
        f"dual_timescale={arch_flags['use_dual_timescale_latent']}, "
        f"direct_ar={arch_flags['use_direct_ar_hybrid']}, "
        f"inlet_imputer={arch_flags['use_inlet_imputer']}, "
        f"nodewise_1d={arch_flags['use_nodewise_1d_dynamics']}, "
        f"stable_rollout={arch_flags['use_stable_flow_rollout']}"
    )

    model_state = load_model_state_dict(args.checkpoint)
    skipped_keys, missing_keys = load_matching_state_dict(model, model_state)
    if skipped_keys:
        print(f"Warning: skipped incompatible checkpoint keys: {len(skipped_keys)} (first 10: {skipped_keys[:10]})")
    if missing_keys:
        print(f"Warning: missing model keys after load: {len(missing_keys)} (first 10: {missing_keys[:10]})")

    model = model.to(device)
    model.eval()
    return model


def write_prediction_rows(
    args,
    model: DualFloodGNN,
    graph,
    norm_stats: Dict,
    device: torch.device,
) -> None:
    events = list_test_events(args.data_dir, args.model_id)
    if args.max_events > 0:
        events = events[:args.max_events]

    out_path = Path(args.output_prediction_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        out_path.unlink()

    writer = None
    total_rows = 0
    try:
        for idx, event_id in enumerate(events, 1):
            event_dir = Path(args.data_dir) / f"Model_{args.model_id}" / "test" / f"event_{event_id}"
            pred_1d, pred_2d = predict_event(
                model=model,
                graph=graph,
                norm_stats=norm_stats,
                event_dir=event_dir,
                device=device,
                prefix_len=args.prefix_len,
                future_inlet_mode=args.future_inlet_mode,
                clamp_min=args.clamp_min,
                clamp_max=args.clamp_max,
            )

            sample_event = load_sample_event_rows(args.sample_submission, args.model_id, event_id)
            pred_rows = map_event_predictions_to_rows(sample_event, pred_1d, pred_2d)

            table = pa.Table.from_pandas(pred_rows, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(str(out_path), table.schema, compression="snappy")
            writer.write_table(table)

            total_rows += len(pred_rows)
            print(
                f"[{idx}/{len(events)}] event_{event_id}: "
                f"rows={len(pred_rows):,}, pred1d_range=({pred_1d.min():.3f},{pred_1d.max():.3f}), "
                f"pred2d_range=({pred_2d.min():.3f},{pred_2d.max():.3f})"
            )
    finally:
        if writer is not None:
            writer.close()

    print(f"Saved model-{args.model_id} prediction rows: {out_path} ({total_rows:,} rows)")


def assemble_submission(args) -> None:
    print(f"Loading base submission: {args.base_submission}")
    base = pd.read_parquet(args.base_submission)
    required_cols = ["row_id", "model_id", "event_id", "node_type", "node_id", "water_level"]
    missing = [c for c in required_cols if c not in base.columns]
    if missing:
        raise ValueError(f"Base submission missing columns: {missing}")
    base = base[required_cols].sort_values("row_id", kind="stable").reset_index(drop=True)

    print(f"Loading model-{args.model_id} prediction rows: {args.output_prediction_rows}")
    pred_rows = pd.read_parquet(args.output_prediction_rows)
    if list(pred_rows.columns) != ["row_id", "water_level"]:
        raise ValueError("Prediction rows must have exactly: row_id, water_level")
    pred_rows = pred_rows.sort_values("row_id", kind="stable").drop_duplicates("row_id", keep="last")

    # Replace rows in-place by row_id index.
    base_idx = base.set_index("row_id", drop=False)
    replace_ids = pred_rows["row_id"].to_numpy(dtype=np.int64)
    base_idx.loc[replace_ids, "water_level"] = pred_rows["water_level"].to_numpy(dtype=np.float32)
    final = base_idx.reset_index(drop=True)

    # Final ordering/schema.
    final = final[required_cols].sort_values("row_id", kind="stable").reset_index(drop=True)

    out_path = Path(args.output_submission)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_parquet(out_path, index=False)

    print(f"Saved final submission: {out_path}")
    print(f"Rows: {len(final):,}")
    print(f"Model 1 range: ({final[final.model_id == 1]['water_level'].min():.3f}, "
          f"{final[final.model_id == 1]['water_level'].max():.3f})")
    print(f"Model 2 range: ({final[final.model_id == 2]['water_level'].min():.3f}, "
          f"{final[final.model_id == 2]['water_level'].max():.3f})")


def parse_args():
    parser = argparse.ArgumentParser(description="Predict DualFlood model and build final submission.")
    parser.add_argument("--model_id", type=int, default=2, help="Competition model id (default: 2)")
    parser.add_argument("--checkpoint", type=str, required=True, help="DualFlood checkpoint path")
    parser.add_argument("--data_dir", type=str, default="./data", help="Dataset root")
    parser.add_argument("--sample_submission", type=str, default="data/sample_submission.parquet",
                        help="Path to sample submission parquet")
    parser.add_argument("--base_submission", type=str, required=True,
                        help="Existing full submission to keep for non-target model rows")
    parser.add_argument("--output_prediction_rows", type=str,
                        default="submission_dualflood_model2_rows.parquet",
                        help="Output row-wise predictions (row_id, water_level)")
    parser.add_argument("--output_submission", type=str, required=True, help="Final submission output parquet")
    parser.add_argument("--device", type=str, default="", help="cuda/cpu; auto when empty")
    parser.add_argument("--prefix_len", type=int, default=10, help="Observed prefix length")
    parser.add_argument(
        "--future_inlet_mode",
        type=str,
        default="missing",
        choices=["observed", "missing", "zero", "last"],
        help="How to provide future inlet_flow during inference",
    )
    parser.add_argument("--pred_len", type=int, default=90, help="Training pred_len used by checkpoint")
    parser.add_argument("--hidden_dim", type=int, default=96, help="DualFlood hidden dim")
    parser.add_argument("--latent_dim", type=int, default=48, help="DualFlood latent dim")
    parser.add_argument("--num_gnn_layers", type=int, default=4, help="DualFlood GNN layers")
    parser.add_argument("--transition_scale", type=float, default=0.05,
                        help="Transition residual step scale used during training")
    parser.add_argument("--coupling_scale", type=float, default=0.03,
                        help="Cross-network coupling scale used during training")
    parser.add_argument("--clamp_min", type=float, default=None, help="Optional minimum clamp for predictions")
    parser.add_argument("--clamp_max", type=float, default=None, help="Optional maximum clamp for predictions")
    parser.add_argument("--max_events", type=int, default=0, help="Debug: run only first N events")
    parser.add_argument("--skip_predict", action="store_true", help="Skip inference and only assemble final file")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("DualFlood Submission Pipeline")
    print(f"Model ID: {args.model_id}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Future inlet mode: {args.future_inlet_mode}")
    print("=" * 70)

    if not args.skip_predict:
        norm_stats = load_norm_stats_from_checkpoint(args.checkpoint)
        static_norm_stats = load_static_norm_stats_from_checkpoint(args.checkpoint)
        if static_norm_stats is None:
            print("Static norm stats missing in checkpoint. Recomputing from training graph...")
            tmp_builder = DualFloodGraphBuilder(args.data_dir, args.model_id)
            static_norm_stats = tmp_builder.compute_static_norm_stats()
        graph = DualFloodGraphBuilder(args.data_dir, args.model_id, static_norm_stats=static_norm_stats).build(split="test")
        model = build_model(args, graph, device)
        write_prediction_rows(args, model, graph, norm_stats, device)
    else:
        print("Skipping prediction generation (using existing row-wise prediction parquet).")

    assemble_submission(args)


if __name__ == "__main__":
    main()
