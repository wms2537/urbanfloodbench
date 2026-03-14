#!/usr/bin/env python3
"""Evaluate a VGSSM checkpoint on the exact seed-42 validation event split.

This uses the same autoregressive rollout path as submission generation, but
targets validation events from the training split instead of test events.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from train_vgssm_standalone import (
    FloodGraphBuilder,
    VGSSM,
    compute_norm_stats_for_events,
    get_model_specific_config,
    infer_model_config_from_state_dict,
    predict_event_autoregressive,
    resolve_internal_output_bounds,
    split_train_val_events,
    strip_model_prefix_from_state_dict,
)


class HoldoutEventDataset:
    """Submission-style event view over a train/val event."""

    def __init__(
        self,
        data_dir: str,
        model_id: int,
        event_id: int,
        split: str,
        norm_stats: Dict[str, Dict[str, np.ndarray]],
        prefix_len: int = 10,
    ) -> None:
        self.data_dir = data_dir
        self.model_id = model_id
        self.event_id = event_id
        self.split = split
        self.norm_stats = norm_stats
        self.prefix_len = prefix_len
        self._load_data()

    def _load_data(self) -> None:
        event_path = Path(self.data_dir) / f"Model_{self.model_id}" / self.split / f"event_{self.event_id}"
        df_1d = pd.read_csv(event_path / "1d_nodes_dynamic_all.csv")
        df_2d = pd.read_csv(event_path / "2d_nodes_dynamic_all.csv")
        df_ts = pd.read_csv(event_path / "timesteps.csv")

        self.timesteps = df_ts["timestep_idx"].to_numpy()
        self.num_timesteps = len(self.timesteps)
        self.num_1d_nodes = int(df_1d["node_idx"].nunique())
        self.num_2d_nodes = int(df_2d["node_idx"].nunique())
        self.node_ids_1d = sorted(df_1d["node_idx"].unique().tolist())
        self.node_ids_2d = sorted(df_2d["node_idx"].unique().tolist())

        self.dynamic_1d = self._reshape_dynamic(df_1d, self.num_1d_nodes, ["water_level", "inlet_flow"])
        self.dynamic_2d = self._reshape_dynamic(df_2d, self.num_2d_nodes, ["rainfall", "water_level", "water_volume"])

        self.prefix_1d = self.dynamic_1d[: self.prefix_len].copy()
        self.prefix_2d = self.dynamic_2d[: self.prefix_len].copy()
        self.future_rainfall = self.dynamic_2d[self.prefix_len :, :, 0:1].copy()
        self.target_1d = self.dynamic_1d[self.prefix_len :, :, 0].copy()
        self.target_2d = self.dynamic_2d[self.prefix_len :, :, 1].copy()
        self.h0_1d = self.dynamic_1d[self.prefix_len - 1 : self.prefix_len, :, 0:1].copy()
        self.h0_2d = self.dynamic_2d[self.prefix_len - 1 : self.prefix_len, :, 1:2].copy()

    def _reshape_dynamic(self, df: pd.DataFrame, num_nodes: int, vars_: List[str]) -> np.ndarray:
        available_vars = [v for v in vars_ if v in df.columns]
        if not available_vars:
            return np.zeros((self.num_timesteps, num_nodes, 1), dtype=np.float32)

        df = df.sort_values(["timestep", "node_idx"])
        num_timesteps = int(df["timestep"].nunique())
        data = df[available_vars].to_numpy(dtype=np.float32, copy=True)
        data = data.reshape(num_timesteps, num_nodes, len(available_vars))
        data = np.nan_to_num(data, nan=0.0)
        return data

    def get_normalized_data(self) -> Dict[str, torch.Tensor]:
        prefix_1d = self.prefix_1d.copy()
        prefix_2d = self.prefix_2d.copy()
        future_rainfall = self.future_rainfall.copy()
        h0_1d = self.h0_1d.copy()
        h0_2d = self.h0_2d.copy()

        if self.norm_stats is not None:
            prefix_1d = (prefix_1d - self.norm_stats["1d"]["mean"]) / self.norm_stats["1d"]["std"]
            prefix_2d = (prefix_2d - self.norm_stats["2d"]["mean"]) / self.norm_stats["2d"]["std"]
            future_rainfall = (future_rainfall - self.norm_stats["2d"]["mean"][0]) / self.norm_stats["2d"]["std"][0]
            h0_1d = (h0_1d - self.norm_stats["target_1d"]["mean"]) / self.norm_stats["target_1d"]["std"]
            h0_2d = (h0_2d - self.norm_stats["target_2d"]["mean"]) / self.norm_stats["target_2d"]["std"]

        return {
            "prefix_1d": torch.from_numpy(prefix_1d).unsqueeze(0),
            "prefix_2d": torch.from_numpy(prefix_2d).unsqueeze(0),
            "future_rainfall": torch.from_numpy(future_rainfall).unsqueeze(0),
            "h0_1d": torch.from_numpy(h0_1d).unsqueeze(0),
            "h0_2d": torch.from_numpy(h0_2d).unsqueeze(0),
        }


def build_runtime_args(cli_args: argparse.Namespace, checkpoint: dict) -> argparse.Namespace:
    hparams = checkpoint.get("hyper_parameters", {}) or {}
    runtime = vars(cli_args).copy()
    runtime.update(hparams)
    runtime["model_id"] = int(cli_args.model_id)
    runtime["data_dir"] = cli_args.data_dir
    runtime["seq_len"] = int(runtime.get("seq_len", 10))
    runtime["stride"] = int(runtime.get("stride", 4))
    raw_prediction_horizon = int(runtime.get("prediction_horizon", 0) or 0)
    cli_prediction_horizon = int(getattr(cli_args, "prediction_horizon", 0) or 0)
    runtime["prediction_horizon"] = raw_prediction_horizon if raw_prediction_horizon > 0 else (cli_prediction_horizon or 90)
    runtime["hidden_dim"] = int(runtime.get("hidden_dim", 64))
    runtime["latent_dim"] = int(runtime.get("latent_dim", 32))
    runtime["event_latent_dim"] = int(runtime.get("event_latent_dim", 16))
    runtime["num_gnn_layers"] = int(runtime.get("num_gnn_layers", 3))
    runtime["num_transition_gnn_layers"] = int(runtime.get("num_transition_gnn_layers", 2))
    runtime["num_heads"] = int(runtime.get("num_heads", 4))
    runtime["dropout"] = float(runtime.get("dropout", 0.2))
    runtime["output_bounds_1d_min"] = runtime.get("output_bounds_1d_min", None)
    runtime["output_bounds_1d_max"] = runtime.get("output_bounds_1d_max", None)
    runtime["output_bounds_2d_min"] = runtime.get("output_bounds_2d_min", None)
    runtime["output_bounds_2d_max"] = runtime.get("output_bounds_2d_max", None)
    runtime["latent_sample_temperature"] = float(runtime.get("latent_sample_temperature", 1.0))
    runtime["latent_state_clip"] = float(runtime.get("latent_state_clip", 10.0))
    return argparse.Namespace(**runtime)


def load_model(
    args: argparse.Namespace,
    checkpoint: dict,
    graph,
    norm_stats: Dict[str, Dict[str, np.ndarray]],
    device: str,
):
    state_dict = strip_model_prefix_from_state_dict(checkpoint["state_dict"])
    model_config = get_model_specific_config(args.model_id, args)
    inferred_config = infer_model_config_from_state_dict(
        state_dict=state_dict,
        fallback_model_config=model_config,
        fallback_event_latent_dim=args.event_latent_dim,
        fallback_num_heads=args.num_heads,
    )

    _, _, output_bounds_1d, output_bounds_2d = resolve_internal_output_bounds(args.model_id, args, norm_stats)

    model = VGSSM(
        static_1d_dim=graph["1d"].x.shape[1],
        static_2d_dim=graph["2d"].x.shape[1],
        dynamic_1d_dim=2,
        dynamic_2d_dim=3,
        hidden_dim=inferred_config["hidden_dim"],
        latent_dim=inferred_config["latent_dim"],
        event_latent_dim=inferred_config["event_latent_dim"],
        num_gnn_layers=inferred_config["num_gnn_layers"],
        num_transition_gnn_layers=inferred_config["num_transition_gnn_layers"],
        num_heads=inferred_config["num_heads"],
        prediction_horizon=args.prediction_horizon,
        use_event_latent=True,
        dropout=inferred_config["dropout"],
        use_delta_prediction=bool(getattr(args, "use_delta_prediction", False)),
        use_physics_loss=bool(getattr(args, "use_physics_loss", False)),
        physics_subsample_rate=int(getattr(args, "physics_subsample_rate", 5)),
        use_timer=bool(getattr(args, "use_timer", False)),
        timer_layers=int(getattr(args, "timer_layers", 4)),
        timer_heads=int(getattr(args, "timer_heads", 4)),
        timer_history_len=int(getattr(args, "timer_history_len", 10)),
        use_timer_v4=bool(getattr(args, "use_timer_v4", False)),
        timer_v4_pooling=str(getattr(args, "timer_v4_pooling", "mean")),
        timer_transition_variant=str(getattr(args, "timer_transition_variant", "auto")),
        timer_enable_2d_context=bool(getattr(args, "timer_enable_2d_context", False)),
        use_grassmann=bool(getattr(args, "use_grassmann", False)),
        grassmann_layers=int(getattr(args, "grassmann_layers", 4)),
        grassmann_rank=int(getattr(args, "grassmann_rank", 12)),
        grassmann_offsets=None,
        use_physics_transition=bool(getattr(args, "use_physics_transition", False)),
        output_bounds_1d=output_bounds_1d,
        output_bounds_2d=output_bounds_2d,
        use_sigmoid_bounds=bool(getattr(args, "use_sigmoid_bounds", False)),
        use_physics_decoder=bool(getattr(args, "use_physics_decoder", False)),
        num_1d_nodes=graph["1d"].x.shape[0],
        num_2d_nodes=graph["2d"].x.shape[0],
        physics_dt=float(getattr(args, "physics_dt", 300.0)),
        latent_sample_temperature=args.latent_sample_temperature,
        latent_state_clip=args.latent_state_clip,
    )

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    return model, inferred_config


def calibrate_event_latent_from_prefix(
    model: VGSSM,
    graph,
    full_prefix_1d: torch.Tensor,
    full_prefix_2d: torch.Tensor,
    target_1d_norm: torch.Tensor,
    target_2d_norm: torch.Tensor,
    future_rainfall_norm: torch.Tensor,
    context_len: int,
    calibration_steps: int,
    calibration_lr: float,
    calibration_reg: float,
    deterministic_latent: bool,
) -> torch.Tensor:
    """Calibrate only the event latent c_e using observed-prefix self-supervision.

    We use the first `context_len` observed timesteps to infer c_e, then optimize
    c_e so the model predicts the remaining observed prefix timesteps. This is
    legal at test time because it only uses the known prefix window.
    """
    context_prefix_1d = full_prefix_1d[:, :context_len]
    context_prefix_2d = full_prefix_2d[:, :context_len]

    with torch.no_grad():
        c_e_init, c_e_mean, _ = model.encode_event_latent(
            context_prefix_1d, context_prefix_2d, deterministic=True
        )

    c_e_opt = torch.nn.Parameter(c_e_mean.clone())
    optimizer = torch.optim.Adam([c_e_opt], lr=calibration_lr)

    model.train()
    for _ in range(calibration_steps):
        optimizer.zero_grad()
        outputs = model(
            graph,
            context_prefix_1d,
            context_prefix_2d,
            prefix_len=context_len,
            future_rainfall=future_rainfall_norm,
            c_e_override=c_e_opt,
            deterministic_latent=deterministic_latent,
        )
        pred_1d = outputs["pred_1d"][:, : target_1d_norm.shape[1], :, 0]
        pred_2d = outputs["pred_2d"][:, : target_2d_norm.shape[1], :, 0]
        loss = F.mse_loss(pred_1d, target_1d_norm) + F.mse_loss(pred_2d, target_2d_norm)
        if calibration_reg > 0:
            loss = loss + calibration_reg * torch.mean((c_e_opt - c_e_init) ** 2)
        loss.backward()
        optimizer.step()

    model.eval()
    return c_e_opt.detach()


def evaluate_checkpoint(args: argparse.Namespace) -> Dict[str, object]:
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    checkpoint = torch.load(args.checkpoint, map_location=device)
    runtime_args = build_runtime_args(args, checkpoint)

    graph = FloodGraphBuilder(runtime_args.data_dir, runtime_args.model_id, add_knn_2d_edges=True, knn_k=8).build(split="train")
    graph = graph.to(device)

    train_events, val_events = split_train_val_events(runtime_args.data_dir, runtime_args.model_id, val_ratio=0.2, seed=42)
    if args.limit_events > 0:
        val_events = val_events[: args.limit_events]

    norm_stats = compute_norm_stats_for_events(
        data_dir=runtime_args.data_dir,
        model_id=runtime_args.model_id,
        event_ids=train_events,
        graph=graph,
        seq_len=runtime_args.seq_len,
        pred_len=runtime_args.prediction_horizon,
        stride=runtime_args.stride,
    )

    model, inferred_config = load_model(runtime_args, checkpoint, graph, norm_stats, device)
    output_bounds_1d_physical, output_bounds_2d_physical, _, _ = resolve_internal_output_bounds(runtime_args.model_id, runtime_args, norm_stats)

    chunk_size = args.chunk_size if args.chunk_size > 0 else int(runtime_args.prediction_horizon)
    deterministic_latent = not bool(args.predict_stochastic_latent)

    sum_sq_1d = 0.0
    sum_sq_2d = 0.0
    sum_err_1d = 0.0
    sum_err_2d = 0.0
    count_1d = 0
    count_2d = 0
    per_t_sq_1d: List[float] = []
    per_t_sq_2d: List[float] = []
    per_t_err_1d: List[float] = []
    per_t_err_2d: List[float] = []
    per_t_c_1d: List[int] = []
    per_t_c_2d: List[int] = []
    event_stats = []

    for event_id in val_events:
        ds = HoldoutEventDataset(runtime_args.data_dir, runtime_args.model_id, event_id, "train", norm_stats, prefix_len=args.prefix_len)
        data = ds.get_normalized_data()
        needed_steps = ds.future_rainfall.shape[0]
        max_steps = needed_steps if args.max_timesteps <= 0 else min(int(args.max_timesteps), int(needed_steps))

        calibrated_c_e = None
        if args.calibrate_event_latent_prefix_only:
            if args.calibration_context_len <= 0 or args.calibration_target_len <= 0:
                raise ValueError("Calibration context/target lengths must be positive")
            if args.calibration_context_len + args.calibration_target_len > args.prefix_len:
                raise ValueError("Calibration context_len + target_len must fit within prefix_len")

            calib_start = int(args.calibration_context_len)
            calib_end = calib_start + int(args.calibration_target_len)
            target_1d_raw = ds.prefix_1d[calib_start:calib_end, :, 0]
            target_2d_raw = ds.prefix_2d[calib_start:calib_end, :, 1]
            target_1d_norm = (
                (target_1d_raw - norm_stats["target_1d"]["mean"]) / norm_stats["target_1d"]["std"]
            )
            target_2d_norm = (
                (target_2d_raw - norm_stats["target_2d"]["mean"]) / norm_stats["target_2d"]["std"]
            )
            target_1d_norm = torch.from_numpy(target_1d_norm).unsqueeze(0).to(device)
            target_2d_norm = torch.from_numpy(target_2d_norm).unsqueeze(0).to(device)
            future_rainfall_norm = data["prefix_2d"][:, calib_start:calib_end, :, 0:1].to(device)
            calibrated_c_e = calibrate_event_latent_from_prefix(
                model=model,
                graph=graph,
                full_prefix_1d=data["prefix_1d"].to(device),
                full_prefix_2d=data["prefix_2d"].to(device),
                target_1d_norm=target_1d_norm,
                target_2d_norm=target_2d_norm,
                future_rainfall_norm=future_rainfall_norm,
                context_len=int(args.calibration_context_len),
                calibration_steps=int(args.calibration_steps),
                calibration_lr=float(args.calibration_lr),
                calibration_reg=float(args.calibration_reg),
                deterministic_latent=deterministic_latent,
            )

        pred_1d, pred_2d = predict_event_autoregressive(
            model,
            graph,
            data,
            norm_stats,
            device=device,
            max_timesteps=max_steps,
            chunk_size=chunk_size,
            deterministic_latent=deterministic_latent,
            stateful_rollout=bool(args.predict_rollout_stateful),
            c_e_override=calibrated_c_e,
        )

        pred_1d = np.clip(pred_1d, output_bounds_1d_physical[0], output_bounds_1d_physical[1])
        pred_2d = np.clip(pred_2d, output_bounds_2d_physical[0], output_bounds_2d_physical[1])

        tgt_1d = ds.target_1d[: pred_1d.shape[0]]
        tgt_2d = ds.target_2d[: pred_2d.shape[0]]

        pred_1d_norm = (pred_1d - norm_stats["target_1d"]["mean"]) / norm_stats["target_1d"]["std"]
        pred_2d_norm = (pred_2d - norm_stats["target_2d"]["mean"]) / norm_stats["target_2d"]["std"]
        tgt_1d_norm = (tgt_1d - norm_stats["target_1d"]["mean"]) / norm_stats["target_1d"]["std"]
        tgt_2d_norm = (tgt_2d - norm_stats["target_2d"]["mean"]) / norm_stats["target_2d"]["std"]

        sq_1d = np.square(pred_1d_norm - tgt_1d_norm)
        sq_2d = np.square(pred_2d_norm - tgt_2d_norm)
        err_1d = pred_1d_norm - tgt_1d_norm
        err_2d = pred_2d_norm - tgt_2d_norm

        sum_sq_1d += float(sq_1d.sum())
        sum_sq_2d += float(sq_2d.sum())
        sum_err_1d += float(err_1d.sum())
        sum_err_2d += float(err_2d.sum())
        count_1d += int(sq_1d.size)
        count_2d += int(sq_2d.size)

        horizon = max(sq_1d.shape[0], sq_2d.shape[0])
        while len(per_t_sq_1d) < horizon:
            per_t_sq_1d.append(0.0)
            per_t_sq_2d.append(0.0)
            per_t_err_1d.append(0.0)
            per_t_err_2d.append(0.0)
            per_t_c_1d.append(0)
            per_t_c_2d.append(0)

        for t in range(sq_1d.shape[0]):
            per_t_sq_1d[t] += float(sq_1d[t].sum())
            per_t_err_1d[t] += float(err_1d[t].sum())
            per_t_c_1d[t] += int(sq_1d[t].size)
        for t in range(sq_2d.shape[0]):
            per_t_sq_2d[t] += float(sq_2d[t].sum())
            per_t_err_2d[t] += float(err_2d[t].sum())
            per_t_c_2d[t] += int(sq_2d[t].size)

        rmse_1d_e = float(np.sqrt(sq_1d.mean()))
        rmse_2d_e = float(np.sqrt(sq_2d.mean()))
        event_stats.append(
            {
                "event_id": int(event_id),
                "std_rmse": 0.5 * (rmse_1d_e + rmse_2d_e),
                "rmse_1d": rmse_1d_e,
                "rmse_2d": rmse_2d_e,
                "steps": int(pred_1d.shape[0]),
            }
        )

    rmse_1d = float(np.sqrt(sum_sq_1d / max(count_1d, 1)))
    rmse_2d = float(np.sqrt(sum_sq_2d / max(count_2d, 1)))
    mean_bias_1d = float(sum_err_1d / max(count_1d, 1))
    mean_bias_2d = float(sum_err_2d / max(count_2d, 1))
    std_rmse = 0.5 * (rmse_1d + rmse_2d)

    per_t_std_rmse = []
    per_t_bias_1d = []
    per_t_bias_2d = []
    for t in range(len(per_t_sq_1d)):
        rmse_t_1d = np.sqrt(per_t_sq_1d[t] / max(per_t_c_1d[t], 1))
        rmse_t_2d = np.sqrt(per_t_sq_2d[t] / max(per_t_c_2d[t], 1))
        per_t_std_rmse.append(0.5 * (float(rmse_t_1d) + float(rmse_t_2d)))
        per_t_bias_1d.append(float(per_t_err_1d[t] / max(per_t_c_1d[t], 1)))
        per_t_bias_2d.append(float(per_t_err_2d[t] / max(per_t_c_2d[t], 1)))

    return {
        "checkpoint": os.path.abspath(args.checkpoint),
        "device": device,
        "model_id": int(runtime_args.model_id),
        "prediction_horizon": int(runtime_args.prediction_horizon),
        "chunk_size": int(chunk_size),
        "prefix_len": int(args.prefix_len),
        "calibrate_event_latent_prefix_only": bool(args.calibrate_event_latent_prefix_only),
        "calibration_context_len": int(args.calibration_context_len),
        "calibration_target_len": int(args.calibration_target_len),
        "calibration_steps": int(args.calibration_steps),
        "calibration_lr": float(args.calibration_lr),
        "calibration_reg": float(args.calibration_reg),
        "num_val_events": int(len(val_events)),
        "val_events": [int(e) for e in val_events],
        "inferred_config": inferred_config,
        "std_rmse": std_rmse,
        "rmse_1d_norm": rmse_1d,
        "rmse_2d_norm": rmse_2d,
        "mean_bias_1d_norm": mean_bias_1d,
        "mean_bias_2d_norm": mean_bias_2d,
        "per_t_std_rmse": per_t_std_rmse,
        "per_t_bias_1d_norm": per_t_bias_1d,
        "per_t_bias_2d_norm": per_t_bias_2d,
        "per_t_sq_sum_1d": per_t_sq_1d,
        "per_t_sq_sum_2d": per_t_sq_2d,
        "per_t_count_1d": per_t_c_1d,
        "per_t_count_2d": per_t_c_2d,
        "event_stats": sorted(event_stats, key=lambda x: x["std_rmse"], reverse=True),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate VGSSM checkpoint on exact validation event split")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--model_id", type=int, required=True, choices=[1, 2])
    parser.add_argument("--prediction_horizon", type=int, default=0, help="Fallback only if absent in checkpoint")
    parser.add_argument("--prefix_len", type=int, default=10)
    parser.add_argument("--chunk_size", type=int, default=0, help="0 uses checkpoint prediction_horizon")
    parser.add_argument("--max_timesteps", type=int, default=0, help="0 evaluates full future event horizon")
    parser.add_argument("--limit_events", type=int, default=0)
    parser.add_argument("--predict_rollout_stateful", action="store_true")
    parser.add_argument("--predict_stochastic_latent", action="store_true")
    parser.add_argument("--calibrate_event_latent_prefix_only", action="store_true")
    parser.add_argument("--calibration_context_len", type=int, default=5)
    parser.add_argument("--calibration_target_len", type=int, default=5)
    parser.add_argument("--calibration_steps", type=int, default=50)
    parser.add_argument("--calibration_lr", type=float, default=0.01)
    parser.add_argument("--calibration_reg", type=float, default=1e-3)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output_json", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_checkpoint(args)
    text = json.dumps(metrics, indent=2)
    print(text)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n")


if __name__ == "__main__":
    main()
