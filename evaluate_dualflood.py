"""
Evaluate DualFlood checkpoints on held-out train events with test-time rollout settings.

This script is intended for scientific debugging of generalization gaps:
- Uses first `prefix_len` timesteps as observed warmup.
- Rolls out autoregressively on the remaining horizon.
- Supports future inlet masking modes used at test time.
- Reports global, horizon-sliced, and per-event standardized RMSE.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from predict_dual_flood import (
    build_model,
    load_norm_stats_from_checkpoint,
    load_static_norm_stats_from_checkpoint,
    reshape_node_dynamic,
)
from train_dual_flood import DualFloodGraphBuilder, DualFloodGNN


def parse_event_ids(event_ids_arg: str) -> Optional[List[int]]:
    s = (event_ids_arg or "").strip()
    if not s:
        return None
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(int(tok))
    return sorted(set(out))


def list_train_events(data_dir: str, model_id: int) -> List[int]:
    train_dir = Path(data_dir) / f"Model_{model_id}" / "train"
    events = []
    for p in train_dir.iterdir():
        name = p.name
        if p.is_dir() and name.startswith("event_"):
            try:
                events.append(int(name.split("_")[1]))
            except (IndexError, ValueError):
                continue
    if not events:
        raise ValueError(f"No train events found under {train_dir}")
    return sorted(events)


def split_events_by_fraction(events: Sequence[int], train_fraction: float) -> Tuple[List[int], List[int]]:
    n_train = max(1, int(len(events) * train_fraction))
    n_train = min(n_train, len(events) - 1) if len(events) > 1 else len(events)
    train_events = list(events[:n_train])
    val_events = list(events[n_train:])
    if not val_events:
        val_events = train_events[-1:]
        train_events = train_events[:-1]
    return train_events, val_events


def choose_events(
    all_events: Sequence[int],
    split: str,
    train_fraction: float,
    explicit_event_ids: Optional[Sequence[int]],
    max_events: int,
) -> List[int]:
    if explicit_event_ids is not None:
        selected = [eid for eid in explicit_event_ids if eid in set(all_events)]
        if not selected:
            raise ValueError("None of --event_ids exist in training events.")
    else:
        train_events, val_events = split_events_by_fraction(all_events, train_fraction)
        if split == "train":
            selected = train_events
        elif split == "val":
            selected = val_events
        elif split == "all":
            selected = list(all_events)
        else:
            raise ValueError(f"Invalid split='{split}'")
    if max_events > 0:
        selected = selected[:max_events]
    if not selected:
        raise ValueError("No events selected for evaluation.")
    return selected


def prepare_event_tensors(
    event_dir: Path,
    norm_stats: Dict,
    prefix_len: int,
    future_inlet_mode: str,
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, np.ndarray]]:
    node_1d = reshape_node_dynamic(event_dir, "1d_nodes_dynamic_all.csv", ["water_level", "inlet_flow"])
    node_2d = reshape_node_dynamic(event_dir, "2d_nodes_dynamic_all.csv", ["water_level", "rainfall", "water_volume"])

    if node_1d.shape[0] != node_2d.shape[0]:
        raise ValueError(f"Timestep mismatch in {event_dir}: {node_1d.shape[0]} vs {node_2d.shape[0]}")

    total_t = int(node_1d.shape[0])
    rollout_len = total_t - prefix_len
    if rollout_len <= 0:
        raise ValueError(f"Invalid rollout length in {event_dir}: total_t={total_t}, prefix_len={prefix_len}")

    n1_mean = norm_stats["node_1d"]["mean"]
    n1_std = norm_stats["node_1d"]["std"]
    n2_mean = norm_stats["node_2d"]["mean"]
    n2_std = norm_stats["node_2d"]["std"]

    input_1d = (node_1d[:prefix_len] - n1_mean.reshape(1, 1, -1)) / n1_std.reshape(1, 1, -1)
    input_2d = (node_2d[:prefix_len] - n2_mean.reshape(1, 1, -1)) / n2_std.reshape(1, 1, -1)

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
        future_inlet = np.full_like(future_inlet, n1_mean[1], dtype=np.float32)
        future_inlet_mask = np.zeros_like(future_inlet_mask, dtype=np.float32)
    elif future_inlet_mode == "observed":
        future_inlet_mask = np.ones_like(future_inlet_mask, dtype=np.float32)
    else:
        raise ValueError(f"Invalid future_inlet_mode='{future_inlet_mode}'")

    future_rainfall = (future_rainfall - n2_mean[1]) / n2_std[1]
    future_inlet = (future_inlet - n1_mean[1]) / n1_std[1]

    # Targets in normalized units for standardized RMSE.
    target_wl_1d = (node_1d[prefix_len:, :, 0] - norm_stats["water_level_1d"]["mean"]) / norm_stats["water_level_1d"]["std"]
    target_wl_2d = (node_2d[prefix_len:, :, 0] - norm_stats["water_level_2d"]["mean"]) / norm_stats["water_level_2d"]["std"]

    tensors = {
        "input_1d": torch.from_numpy(input_1d).unsqueeze(0).to(device=device, dtype=torch.float32),
        "input_2d": torch.from_numpy(input_2d).unsqueeze(0).to(device=device, dtype=torch.float32),
        "future_rainfall": torch.from_numpy(future_rainfall).unsqueeze(0).to(device=device, dtype=torch.float32),
        "future_inlet": torch.from_numpy(future_inlet).unsqueeze(0).to(device=device, dtype=torch.float32),
        "future_inlet_mask": torch.from_numpy(future_inlet_mask).unsqueeze(0).to(device=device, dtype=torch.float32),
    }
    arrays = {
        "target_wl_1d_norm": target_wl_1d.astype(np.float32),
        "target_wl_2d_norm": target_wl_2d.astype(np.float32),
        "rainfall_future_raw": node_2d[prefix_len:, :, 1].astype(np.float32),
        "rollout_len": np.int64(rollout_len),
    }
    return tensors, arrays


def ensure_len(buf: List[float], new_len: int, fill_value: float = 0.0) -> None:
    if len(buf) < new_len:
        buf.extend([fill_value] * (new_len - len(buf)))


def evaluate(
    model: DualFloodGNN,
    graph,
    events: Sequence[int],
    data_dir: str,
    model_id: int,
    norm_stats: Dict,
    prefix_len: int,
    future_inlet_mode: str,
    device: torch.device,
) -> Dict:
    sum_sq_1d = 0.0
    sum_sq_2d = 0.0
    count_1d = 0
    count_2d = 0

    per_t_sq_1d: List[float] = []
    per_t_sq_2d: List[float] = []
    per_t_c_1d: List[int] = []
    per_t_c_2d: List[int] = []

    event_stats = []

    for idx, eid in enumerate(events, 1):
        event_dir = Path(data_dir) / f"Model_{model_id}" / "train" / f"event_{eid}"
        tensors, arrays = prepare_event_tensors(
            event_dir=event_dir,
            norm_stats=norm_stats,
            prefix_len=prefix_len,
            future_inlet_mode=future_inlet_mode,
            device=device,
        )
        rollout_len = int(arrays["rollout_len"])

        with torch.no_grad():
            outputs = model(
                graph=graph,
                input_1d=tensors["input_1d"],
                input_2d=tensors["input_2d"],
                future_rainfall=tensors["future_rainfall"],
                future_inlet=tensors["future_inlet"],
                future_inlet_mask=tensors["future_inlet_mask"],
                rollout_len=rollout_len,
            )

        pred_1d = outputs["pred_wl_1d"][0, :rollout_len, :, 0].detach().cpu().numpy().astype(np.float32)
        pred_2d = outputs["pred_wl_2d"][0, :rollout_len, :, 0].detach().cpu().numpy().astype(np.float32)

        target_1d = arrays["target_wl_1d_norm"][:rollout_len]
        target_2d = arrays["target_wl_2d_norm"][:rollout_len]

        sq_1d = (pred_1d - target_1d) ** 2
        sq_2d = (pred_2d - target_2d) ** 2

        sum_sq_1d += float(sq_1d.sum())
        sum_sq_2d += float(sq_2d.sum())
        count_1d += int(sq_1d.size)
        count_2d += int(sq_2d.size)

        ensure_len(per_t_sq_1d, rollout_len)
        ensure_len(per_t_sq_2d, rollout_len)
        ensure_len(per_t_c_1d, rollout_len, 0)
        ensure_len(per_t_c_2d, rollout_len, 0)
        for t in range(rollout_len):
            per_t_sq_1d[t] += float(sq_1d[t].sum())
            per_t_sq_2d[t] += float(sq_2d[t].sum())
            per_t_c_1d[t] += int(sq_1d[t].size)
            per_t_c_2d[t] += int(sq_2d[t].size)

        rmse_1d_e = float(np.sqrt(sq_1d.mean()))
        rmse_2d_e = float(np.sqrt(sq_2d.mean()))
        std_rmse_e = 0.5 * (rmse_1d_e + rmse_2d_e)

        rainfall_future = arrays["rainfall_future_raw"]
        rainfall_mean = float(np.mean(rainfall_future))
        rainfall_max = float(np.max(rainfall_future))
        rainfall_sum = float(np.sum(rainfall_future))

        event_stats.append(
            {
                "event_id": int(eid),
                "std_rmse": std_rmse_e,
                "rmse_1d": rmse_1d_e,
                "rmse_2d": rmse_2d_e,
                "rollout_len": int(rollout_len),
                "rainfall_mean": rainfall_mean,
                "rainfall_max": rainfall_max,
                "rainfall_sum": rainfall_sum,
            }
        )

        print(
            f"[{idx}/{len(events)}] event_{eid}: "
            f"std_rmse={std_rmse_e:.6f} "
            f"(1d={rmse_1d_e:.6f}, 2d={rmse_2d_e:.6f}) "
            f"rain_mean={rainfall_mean:.6f} rain_max={rainfall_max:.6f}"
        )

    rmse_1d = float(np.sqrt(sum_sq_1d / max(count_1d, 1)))
    rmse_2d = float(np.sqrt(sum_sq_2d / max(count_2d, 1)))
    std_rmse = 0.5 * (rmse_1d + rmse_2d)

    per_t_std_rmse = []
    for t in range(len(per_t_sq_1d)):
        rmse_t_1d = np.sqrt(per_t_sq_1d[t] / max(per_t_c_1d[t], 1))
        rmse_t_2d = np.sqrt(per_t_sq_2d[t] / max(per_t_c_2d[t], 1))
        per_t_std_rmse.append(0.5 * (float(rmse_t_1d) + float(rmse_t_2d)))

    segments = []
    max_t = len(per_t_std_rmse)
    bounds = [(1, 30), (31, 90), (91, 180), (181, 300), (301, max_t)]
    for lo, hi in bounds:
        if lo > max_t:
            continue
        hi = min(hi, max_t)
        vals = per_t_std_rmse[lo - 1:hi]
        if not vals:
            continue
        segments.append(
            {
                "start_t": int(lo),
                "end_t": int(hi),
                "mean_std_rmse": float(np.mean(vals)),
            }
        )

    event_stats_sorted = sorted(event_stats, key=lambda x: x["std_rmse"], reverse=True)
    return {
        "num_events": int(len(events)),
        "events": [int(e) for e in events],
        "global": {
            "std_rmse": std_rmse,
            "rmse_1d_norm": rmse_1d,
            "rmse_2d_norm": rmse_2d,
        },
        "horizon": {
            "per_t_std_rmse": per_t_std_rmse,
            "segments": segments,
        },
        "event_stats": event_stats_sorted,
    }


def build_model_from_checkpoint(
    checkpoint: str,
    data_dir: str,
    model_id: int,
    device: torch.device,
    pred_len_hint: int,
    hidden_dim: int,
    latent_dim: int,
    num_gnn_layers: int,
    transition_scale: float,
    coupling_scale: float,
):
    static_norm_stats = load_static_norm_stats_from_checkpoint(checkpoint)
    if static_norm_stats is None:
        print("Static norm stats missing in checkpoint. Recomputing from training graph...")
        tmp_builder = DualFloodGraphBuilder(data_dir, model_id)
        static_norm_stats = tmp_builder.compute_static_norm_stats()
    graph = DualFloodGraphBuilder(data_dir, model_id, static_norm_stats=static_norm_stats).build(split="train")

    class Args:
        pass

    args = Args()
    args.checkpoint = checkpoint
    args.pred_len = pred_len_hint
    args.hidden_dim = hidden_dim
    args.latent_dim = latent_dim
    args.num_gnn_layers = num_gnn_layers
    args.transition_scale = transition_scale
    args.coupling_scale = coupling_scale

    graph = graph.to(device)
    model = build_model(args, graph, device)
    model.eval()
    return model, graph


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DualFlood checkpoint on held-out train events.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--model_id", type=int, default=2, help="Competition model id")
    parser.add_argument("--data_dir", type=str, default="./data", help="Dataset root")
    parser.add_argument("--device", type=str, default="", help="cuda/cpu; auto when empty")
    parser.add_argument("--prefix_len", type=int, default=10, help="Observed warmup length")
    parser.add_argument(
        "--future_inlet_mode",
        type=str,
        default="missing",
        choices=["observed", "missing", "zero", "last"],
        help="How to provide future inlet during rollout",
    )
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "all"], help="Event split")
    parser.add_argument("--train_fraction", type=float, default=0.8, help="Train split fraction used for val partition")
    parser.add_argument("--event_ids", type=str, default="", help="Comma-separated explicit event ids")
    parser.add_argument("--max_events", type=int, default=0, help="Evaluate only first N selected events")
    parser.add_argument("--pred_len_hint", type=int, default=90, help="Model constructor pred_len hint")
    parser.add_argument("--hidden_dim", type=int, default=96, help="Model hidden dim")
    parser.add_argument("--latent_dim", type=int, default=48, help="Model latent dim")
    parser.add_argument("--num_gnn_layers", type=int, default=4, help="Model GNN layers")
    parser.add_argument("--transition_scale", type=float, default=0.05, help="Transition scale")
    parser.add_argument("--coupling_scale", type=float, default=0.03, help="Coupling scale")
    parser.add_argument("--output_json", type=str, default="", help="Optional output JSON path")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")

    all_events = list_train_events(args.data_dir, args.model_id)
    explicit = parse_event_ids(args.event_ids)
    events = choose_events(
        all_events=all_events,
        split=args.split,
        train_fraction=args.train_fraction,
        explicit_event_ids=explicit,
        max_events=args.max_events,
    )

    print("=" * 70)
    print("DualFlood Checkpoint Evaluation")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Model ID: {args.model_id}")
    print(f"Split: {args.split}")
    print(f"Events: {events}")
    print(f"Future inlet mode: {args.future_inlet_mode}")
    print("=" * 70)

    norm_stats = load_norm_stats_from_checkpoint(args.checkpoint)
    model, graph = build_model_from_checkpoint(
        checkpoint=args.checkpoint,
        data_dir=args.data_dir,
        model_id=args.model_id,
        device=device,
        pred_len_hint=args.pred_len_hint,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_gnn_layers=args.num_gnn_layers,
        transition_scale=args.transition_scale,
        coupling_scale=args.coupling_scale,
    )

    summary = evaluate(
        model=model,
        graph=graph,
        events=events,
        data_dir=args.data_dir,
        model_id=args.model_id,
        norm_stats=norm_stats,
        prefix_len=args.prefix_len,
        future_inlet_mode=args.future_inlet_mode,
        device=device,
    )

    print("=" * 70)
    print("Global Metrics")
    print(json.dumps(summary["global"], indent=2))
    print("Horizon Segments")
    print(json.dumps(summary["horizon"]["segments"], indent=2))
    print("Worst 5 Events")
    print(json.dumps(summary["event_stats"][:5], indent=2))
    print("=" * 70)

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2))
        print(f"Saved summary JSON: {out_path}")


if __name__ == "__main__":
    main()
