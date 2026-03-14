#!/usr/bin/env python3
"""
Fit simple ARX baselines for both competition models and build a submission.

Model choices (scientific baseline):
- Model 1: shared ARX(1) with rainfall forcing
- Model 2: node-wise ARX(1) with rainfall forcing

Workflow:
1) Optional holdout validation on train split (80/20 by event id order)
2) Fit final ARX models on all train events
3) Generate row-wise predictions for ALL test rows
4) Assemble final submission parquet in official schema/order
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def list_events(data_dir: Path, model_id: int, split: str) -> List[int]:
    split_dir = data_dir / f"Model_{model_id}" / split
    events = sorted(
        int(p.name.split("_")[1])
        for p in split_dir.iterdir()
        if p.is_dir() and p.name.startswith("event_")
    )
    if not events:
        raise ValueError(f"No events found under: {split_dir}")
    return events


def load_event_wl_rain(event_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    df1 = pd.read_csv(
        event_dir / "1d_nodes_dynamic_all.csv",
        usecols=["timestep", "node_idx", "water_level"],
    ).sort_values(["timestep", "node_idx"])
    df2 = pd.read_csv(
        event_dir / "2d_nodes_dynamic_all.csv",
        usecols=["timestep", "node_idx", "water_level", "rainfall"],
    ).sort_values(["timestep", "node_idx"])

    t1 = int(df1["timestep"].nunique())
    t2 = int(df2["timestep"].nunique())
    n1 = int(df1["node_idx"].nunique())
    n2 = int(df2["node_idx"].nunique())
    if t1 != t2:
        raise ValueError(f"Timestep mismatch in {event_dir}: {t1} vs {t2}")

    wl1 = df1["water_level"].to_numpy(dtype=np.float64).reshape(t1, n1)
    wl2 = df2["water_level"].to_numpy(dtype=np.float64).reshape(t2, n2)
    rain2 = df2["rainfall"].to_numpy(dtype=np.float64).reshape(t2, n2)
    return wl1, wl2, rain2


def load_event_wl2_rain(event_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    df2 = pd.read_csv(
        event_dir / "2d_nodes_dynamic_all.csv",
        usecols=["timestep", "node_idx", "water_level", "rainfall"],
    ).sort_values(["timestep", "node_idx"])
    t2 = int(df2["timestep"].nunique())
    n2 = int(df2["node_idx"].nunique())
    wl2 = df2["water_level"].to_numpy(dtype=np.float64).reshape(t2, n2)
    rain2 = df2["rainfall"].to_numpy(dtype=np.float64).reshape(t2, n2)
    return wl2, rain2


def split_events(events: Sequence[int], train_fraction: float) -> Tuple[List[int], List[int]]:
    n = len(events)
    n_train = max(1, int(n * train_fraction))
    n_train = min(n_train, n - 1) if n > 1 else n
    train_events = list(events[:n_train])
    val_events = list(events[n_train:])
    if not val_events:
        val_events = train_events[-1:]
        train_events = train_events[:-1]
    return train_events, val_events


def compute_target_stds(data_dir: Path, model_id: int, train_events: Sequence[int]) -> Tuple[float, float]:
    all1 = []
    all2 = []
    for eid in train_events:
        event_dir = data_dir / f"Model_{model_id}" / "train" / f"event_{eid}"
        wl1, wl2, _ = load_event_wl_rain(event_dir)
        all1.append(wl1.reshape(-1))
        all2.append(wl2.reshape(-1))
    vec1 = np.concatenate(all1)
    vec2 = np.concatenate(all2)
    std1 = float(vec1.std() + 1e-8)
    std2 = float(vec2.std() + 1e-8)
    return std1, std2


@dataclass
class SharedARX:
    # y_{t+1} = a*y_t + b*r_t + c
    w1: np.ndarray  # shape (3,)
    w2: np.ndarray  # shape (3,)

    def predict_rollout(
        self,
        wl1_prev: np.ndarray,
        wl2_prev: np.ndarray,
        rain_forcing: np.ndarray,  # [H, N2], uses mean for 1D and local for 2D
    ) -> Tuple[np.ndarray, np.ndarray]:
        h = int(rain_forcing.shape[0])
        n1 = int(wl1_prev.shape[0])
        n2 = int(wl2_prev.shape[0])
        out1 = np.zeros((h, n1), dtype=np.float64)
        out2 = np.zeros((h, n2), dtype=np.float64)

        p1 = wl1_prev.copy()
        p2 = wl2_prev.copy()
        for t in range(h):
            r_mean = float(rain_forcing[t].mean())
            p1 = self.w1[0] * p1 + self.w1[1] * r_mean + self.w1[2]
            p2 = self.w2[0] * p2 + self.w2[1] * rain_forcing[t] + self.w2[2]
            out1[t] = p1
            out2[t] = p2
        return out1, out2


@dataclass
class NodewiseARX:
    # y_{t+1,n} = a_n*y_{t,n} + b_n*r_t + c_n
    w1: np.ndarray  # shape (N1, 3)
    w2: np.ndarray  # shape (N2, 3)

    def predict_rollout(
        self,
        wl1_prev: np.ndarray,
        wl2_prev: np.ndarray,
        rain_forcing: np.ndarray,  # [H, N2], uses global rainfall
    ) -> Tuple[np.ndarray, np.ndarray]:
        h = int(rain_forcing.shape[0])
        n1 = int(wl1_prev.shape[0])
        n2 = int(wl2_prev.shape[0])
        out1 = np.zeros((h, n1), dtype=np.float64)
        out2 = np.zeros((h, n2), dtype=np.float64)

        p1 = wl1_prev.copy()
        p2 = wl2_prev.copy()
        for t in range(h):
            # Model 2 rainfall is uniform across 2D nodes; mean is robust.
            rt = float(rain_forcing[t].mean())
            p1 = self.w1[:, 0] * p1 + self.w1[:, 1] * rt + self.w1[:, 2]
            p2 = self.w2[:, 0] * p2 + self.w2[:, 1] * rt + self.w2[:, 2]
            out1[t] = p1
            out2[t] = p2
        return out1, out2


@dataclass
class RainRetrievalShiftBank:
    """
    Event retrieval bank for Model 2.

    Retrieval key:
    - forcing rainfall sequence over rollout horizon (global rainfall per timestep)

    Prediction:
    - nearest (or top-k weighted) historical wl2 trajectory
    - shifted by node-wise warmup endpoint offset
    """

    rain_forcing_bank: List[np.ndarray]  # len=E, each [Hi]
    wl1_start_bank: np.ndarray  # [E, N1], timestep prefix_len-1
    wl2_start_bank: np.ndarray  # [E, N2], timestep prefix_len-1
    wl1_future_bank: List[np.ndarray]  # len=E, each [Hi, N1]
    wl2_future_bank: List[np.ndarray]  # len=E, each [Hi, N2]
    _cache: Dict[int, Tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None

    @staticmethod
    def _time_resample_index(src_len: int, dst_len: int) -> np.ndarray:
        if src_len <= 0 or dst_len <= 0:
            raise ValueError(f"Invalid resample lengths: src={src_len}, dst={dst_len}")
        if src_len == dst_len:
            return np.arange(src_len, dtype=np.int32)
        return np.round(np.linspace(0, src_len - 1, dst_len)).astype(np.int32)

    def _get_resampled_bank(self, horizon: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._cache is None:
            self._cache = {}
        cached = self._cache.get(int(horizon))
        if cached is not None:
            return cached

        rain_rows: List[np.ndarray] = []
        wl1_rows: List[np.ndarray] = []
        wl2_rows: List[np.ndarray] = []
        for rain_i, wl1_i, wl2_i in zip(self.rain_forcing_bank, self.wl1_future_bank, self.wl2_future_bank):
            idx = self._time_resample_index(src_len=int(rain_i.shape[0]), dst_len=int(horizon))
            rain_rows.append(rain_i[idx])
            wl1_rows.append(wl1_i[idx])
            wl2_rows.append(wl2_i[idx])

        rain_bank_h = np.stack(rain_rows, axis=0)  # [E, H]
        wl1_bank_h = np.stack(wl1_rows, axis=0)  # [E, H, N1]
        wl2_bank_h = np.stack(wl2_rows, axis=0)  # [E, H, N2]
        self._cache[int(horizon)] = (rain_bank_h, wl1_bank_h, wl2_bank_h)
        return rain_bank_h, wl1_bank_h, wl2_bank_h

    def predict_wl1_wl2(
        self,
        wl1_prev: np.ndarray,  # [N1], observed at prefix_len-1
        wl2_prev: np.ndarray,  # [N2], observed at prefix_len-1
        rain_forcing: np.ndarray,  # [H, N2]
        top_k: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        rain_global = rain_forcing.mean(axis=1)  # [H]
        horizon = int(rain_global.shape[0])
        rain_bank_h, wl1_bank_h, wl2_bank_h = self._get_resampled_bank(horizon)

        d = np.mean((rain_bank_h - rain_global[None, :]) ** 2, axis=1)
        k = max(1, min(int(top_k), int(d.shape[0])))
        if k == 1:
            idx = int(np.argmin(d))
            shift1 = wl1_prev - self.wl1_start_bank[idx]
            shift = wl2_prev - self.wl2_start_bank[idx]
            return wl1_bank_h[idx] + shift1[None, :], wl2_bank_h[idx] + shift[None, :]

        top_idx = np.argpartition(d, k - 1)[:k]
        top_d = d[top_idx]
        w = 1.0 / (top_d + 1e-8)
        w = w / w.sum()
        shifts1 = wl1_prev[None, :] - self.wl1_start_bank[top_idx]  # [k, N1]
        shifts = wl2_prev[None, :] - self.wl2_start_bank[top_idx]  # [k, N2]
        cands1 = wl1_bank_h[top_idx] + shifts1[:, None, :]  # [k, H, N1]
        cands = wl2_bank_h[top_idx] + shifts[:, None, :]  # [k, H, N2]
        pred1 = np.tensordot(w, cands1, axes=(0, 0))
        pred2 = np.tensordot(w, cands, axes=(0, 0))
        return pred1, pred2


@dataclass
class BlendedModel2:
    arx: NodewiseARX
    retrieval_bank: RainRetrievalShiftBank
    arx_weight_1d: float = 0.5
    arx_weight_2d: float = 0.5
    retrieval_top_k: int = 1

    def predict_rollout(
        self,
        wl1_prev: np.ndarray,
        wl2_prev: np.ndarray,
        rain_forcing: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        pred1_arx, pred2_arx = self.arx.predict_rollout(
            wl1_prev=wl1_prev,
            wl2_prev=wl2_prev,
            rain_forcing=rain_forcing,
        )
        pred1_ret, pred2_ret = self.retrieval_bank.predict_wl1_wl2(
            wl1_prev=wl1_prev,
            wl2_prev=wl2_prev,
            rain_forcing=rain_forcing,
            top_k=self.retrieval_top_k,
        )
        w1 = float(np.clip(self.arx_weight_1d, 0.0, 1.0))
        w2 = float(np.clip(self.arx_weight_2d, 0.0, 1.0))
        pred1 = w1 * pred1_arx + (1.0 - w1) * pred1_ret
        pred2 = w2 * pred2_arx + (1.0 - w2) * pred2_ret
        return pred1, pred2


def fit_shared_arx(data_dir: Path, model_id: int, train_events: Sequence[int]) -> SharedARX:
    p = 3
    xtx1 = np.zeros((p, p), dtype=np.float64)
    xty1 = np.zeros((p,), dtype=np.float64)
    xtx2 = np.zeros((p, p), dtype=np.float64)
    xty2 = np.zeros((p,), dtype=np.float64)

    for eid in train_events:
        event_dir = data_dir / f"Model_{model_id}" / "train" / f"event_{eid}"
        wl1, wl2, rain2 = load_event_wl_rain(event_dir)
        rain_mean = rain2.mean(axis=1)

        # 1D shared
        x_1d = wl1[:-1]
        y_1d = wl1[1:]
        r_1d = np.repeat(rain_mean[:-1, None], x_1d.shape[1], axis=1)
        x = np.stack([x_1d, r_1d, np.ones_like(x_1d)], axis=-1).reshape(-1, p)
        y = y_1d.reshape(-1)
        xtx1 += x.T @ x
        xty1 += x.T @ y

        # 2D shared (local rainfall)
        x_2d = wl2[:-1]
        y_2d = wl2[1:]
        r_2d = rain2[:-1]
        x = np.stack([x_2d, r_2d, np.ones_like(x_2d)], axis=-1).reshape(-1, p)
        y = y_2d.reshape(-1)
        xtx2 += x.T @ x
        xty2 += x.T @ y

    lam = 1e-6
    w1 = np.linalg.solve(xtx1 + lam * np.eye(p), xty1)
    w2 = np.linalg.solve(xtx2 + lam * np.eye(p), xty2)
    return SharedARX(w1=w1, w2=w2)


def _init_nodewise_acc(n_nodes: int) -> Dict[str, np.ndarray]:
    return {
        "s11": np.zeros(n_nodes, dtype=np.float64),
        "s12": np.zeros(n_nodes, dtype=np.float64),
        "s13": np.zeros(n_nodes, dtype=np.float64),
        "s22": np.zeros(n_nodes, dtype=np.float64),
        "s23": np.zeros(n_nodes, dtype=np.float64),
        "s33": np.zeros(n_nodes, dtype=np.float64),
        "t1": np.zeros(n_nodes, dtype=np.float64),
        "t2": np.zeros(n_nodes, dtype=np.float64),
        "t3": np.zeros(n_nodes, dtype=np.float64),
    }


def _solve_nodewise_arx(acc: Dict[str, np.ndarray], lam: float) -> np.ndarray:
    n = int(acc["s11"].shape[0])
    out = np.zeros((n, 3), dtype=np.float64)
    eye = np.eye(3, dtype=np.float64)
    for i in range(n):
        m = np.array(
            [
                [acc["s11"][i], acc["s12"][i], acc["s13"][i]],
                [acc["s12"][i], acc["s22"][i], acc["s23"][i]],
                [acc["s13"][i], acc["s23"][i], acc["s33"][i]],
            ],
            dtype=np.float64,
        )
        b = np.array([acc["t1"][i], acc["t2"][i], acc["t3"][i]], dtype=np.float64)
        out[i] = np.linalg.solve(m + lam * eye, b)
    return out


def fit_nodewise_arx_model2(data_dir: Path, train_events: Sequence[int]) -> NodewiseARX:
    first = data_dir / "Model_2" / "train" / f"event_{train_events[0]}"
    n1 = int(pd.read_csv(first / "1d_nodes_dynamic_all.csv", usecols=["node_idx"])["node_idx"].nunique())
    n2 = int(pd.read_csv(first / "2d_nodes_dynamic_all.csv", usecols=["node_idx"])["node_idx"].nunique())
    acc1 = _init_nodewise_acc(n1)
    acc2 = _init_nodewise_acc(n2)

    for eid in train_events:
        event_dir = data_dir / "Model_2" / "train" / f"event_{eid}"
        wl1, wl2, rain2 = load_event_wl_rain(event_dir)
        rain_global = rain2[:, 0]

        x1 = wl1[:-1]
        y1 = wl1[1:]
        r1 = np.repeat(rain_global[:-1, None], n1, axis=1)
        acc1["s11"] += np.sum(x1 * x1, axis=0)
        acc1["s12"] += np.sum(x1 * r1, axis=0)
        acc1["s13"] += np.sum(x1, axis=0)
        acc1["s22"] += np.sum(r1 * r1, axis=0)
        acc1["s23"] += np.sum(r1, axis=0)
        acc1["s33"] += x1.shape[0]
        acc1["t1"] += np.sum(x1 * y1, axis=0)
        acc1["t2"] += np.sum(r1 * y1, axis=0)
        acc1["t3"] += np.sum(y1, axis=0)

        x2 = wl2[:-1]
        y2 = wl2[1:]
        r2 = np.repeat(rain_global[:-1, None], n2, axis=1)
        acc2["s11"] += np.sum(x2 * x2, axis=0)
        acc2["s12"] += np.sum(x2 * r2, axis=0)
        acc2["s13"] += np.sum(x2, axis=0)
        acc2["s22"] += np.sum(r2 * r2, axis=0)
        acc2["s23"] += np.sum(r2, axis=0)
        acc2["s33"] += x2.shape[0]
        acc2["t1"] += np.sum(x2 * y2, axis=0)
        acc2["t2"] += np.sum(r2 * y2, axis=0)
        acc2["t3"] += np.sum(y2, axis=0)

    lam = 1e-5
    w1 = _solve_nodewise_arx(acc1, lam=lam)
    w2 = _solve_nodewise_arx(acc2, lam=lam)
    return NodewiseARX(w1=w1, w2=w2)


def build_model2_retrieval_bank(
    data_dir: Path,
    train_events: Sequence[int],
    prefix_len: int,
) -> RainRetrievalShiftBank:
    rain_bank: List[np.ndarray] = []
    wl1_start_bank: List[np.ndarray] = []
    wl2_start_bank: List[np.ndarray] = []
    wl1_future_bank: List[np.ndarray] = []
    wl2_future_bank: List[np.ndarray] = []

    for eid in train_events:
        event_dir = data_dir / "Model_2" / "train" / f"event_{eid}"
        wl1, wl2, rain2 = load_event_wl_rain(event_dir)
        if wl1.shape[0] <= prefix_len:
            raise ValueError(f"Event too short for retrieval bank: event_{eid}")
        rain_forcing = rain2[prefix_len - 1 : -1].mean(axis=1)  # [H]
        wl1_start = wl1[prefix_len - 1]  # [N1]
        wl2_start = wl2[prefix_len - 1]  # [N2]
        wl1_future = wl1[prefix_len:]  # [H, N1]
        wl2_future = wl2[prefix_len:]  # [H, N2]
        if rain_forcing.shape[0] != wl1_future.shape[0] or rain_forcing.shape[0] != wl2_future.shape[0]:
            raise ValueError(
                f"Bank forcing/target mismatch for event_{eid}: "
                f"{rain_forcing.shape[0]} vs {wl2_future.shape[0]}"
            )
        rain_bank.append(rain_forcing.astype(np.float64, copy=False))
        wl1_start_bank.append(wl1_start.astype(np.float64, copy=False))
        wl2_start_bank.append(wl2_start.astype(np.float64, copy=False))
        wl1_future_bank.append(wl1_future.astype(np.float64, copy=False))
        wl2_future_bank.append(wl2_future.astype(np.float64, copy=False))

    return RainRetrievalShiftBank(
        rain_forcing_bank=rain_bank,
        wl1_start_bank=np.stack(wl1_start_bank, axis=0),
        wl2_start_bank=np.stack(wl2_start_bank, axis=0),
        wl1_future_bank=wl1_future_bank,
        wl2_future_bank=wl2_future_bank,
    )


def evaluate_arx_model(
    data_dir: Path,
    model_id: int,
    model,
    val_events: Sequence[int],
    prefix_len: int,
    std_1d: float,
    std_2d: float,
) -> Dict[str, float]:
    se1 = 0.0
    se2 = 0.0
    c1 = 0
    c2 = 0

    for eid in val_events:
        event_dir = data_dir / f"Model_{model_id}" / "train" / f"event_{eid}"
        wl1, wl2, rain2 = load_event_wl_rain(event_dir)
        horizon = int(wl1.shape[0] - prefix_len)
        if horizon <= 0:
            continue
        rain_forcing = rain2[prefix_len - 1 : -1]  # length == horizon
        pred1, pred2 = model.predict_rollout(
            wl1_prev=wl1[prefix_len - 1],
            wl2_prev=wl2[prefix_len - 1],
            rain_forcing=rain_forcing,
        )
        target1 = wl1[prefix_len:]
        target2 = wl2[prefix_len:]
        e1 = ((pred1 - target1) / std_1d) ** 2
        e2 = ((pred2 - target2) / std_2d) ** 2
        se1 += float(e1.sum())
        se2 += float(e2.sum())
        c1 += int(e1.size)
        c2 += int(e2.size)

    rm1 = float(np.sqrt(se1 / max(c1, 1)))
    rm2 = float(np.sqrt(se2 / max(c2, 1)))
    return {
        "std_rmse": 0.5 * (rm1 + rm2),
        "rmse_1d_norm": rm1,
        "rmse_2d_norm": rm2,
    }


def load_sample_event_rows(sample_submission: Path, model_id: int, event_id: int) -> pd.DataFrame:
    df = pd.read_parquet(
        sample_submission,
        columns=["row_id", "model_id", "event_id", "node_type", "node_id"],
        filters=[("model_id", "==", model_id), ("event_id", "==", event_id)],
    )
    df = df.sort_values("row_id", kind="stable").reset_index(drop=True)
    df["timestep"] = df.groupby(["node_type", "node_id"]).cumcount().astype(np.int32)
    return df


def map_event_predictions_to_rows(sample_event: pd.DataFrame, pred_1d: np.ndarray, pred_2d: np.ndarray) -> pd.DataFrame:
    node_type = sample_event["node_type"].to_numpy(dtype=np.int16)
    node_id = sample_event["node_id"].to_numpy(dtype=np.int32)
    timestep = sample_event["timestep"].to_numpy(dtype=np.int32)
    out = np.empty(len(sample_event), dtype=np.float32)

    mask1 = node_type == 1
    mask2 = node_type == 2
    if mask1.any():
        out[mask1] = pred_1d[timestep[mask1], node_id[mask1]]
    if mask2.any():
        out[mask2] = pred_2d[timestep[mask2], node_id[mask2]]

    return pd.DataFrame(
        {
            "row_id": sample_event["row_id"].to_numpy(dtype=np.int64),
            "water_level": out,
        }
    )


def write_prediction_rows(
    data_dir: Path,
    sample_submission: Path,
    model_1,
    model_2,
    prefix_len: int,
    output_prediction_rows: Path,
) -> int:
    if output_prediction_rows.exists():
        output_prediction_rows.unlink()
    output_prediction_rows.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    total_rows = 0
    try:
        for model_id, model in ((1, model_1), (2, model_2)):
            test_events = list_events(data_dir, model_id, "test")
            for idx, eid in enumerate(test_events, 1):
                event_dir = data_dir / f"Model_{model_id}" / "test" / f"event_{eid}"
                wl1, wl2, rain2 = load_event_wl_rain(event_dir)
                horizon = int(wl1.shape[0] - prefix_len)
                if horizon <= 0:
                    raise ValueError(f"Invalid horizon for model={model_id} event={eid}")
                rain_forcing = rain2[prefix_len - 1 : -1]
                if rain_forcing.shape[0] != horizon:
                    raise ValueError(
                        f"Forcing length mismatch for model={model_id} event={eid}: "
                        f"{rain_forcing.shape[0]} vs horizon={horizon}"
                    )

                pred1, pred2 = model.predict_rollout(
                    wl1_prev=wl1[prefix_len - 1],
                    wl2_prev=wl2[prefix_len - 1],
                    rain_forcing=rain_forcing,
                )
                if not np.isfinite(pred1).all() or not np.isfinite(pred2).all():
                    raise ValueError(f"Non-finite predictions for model={model_id} event={eid}")

                sample_event = load_sample_event_rows(sample_submission, model_id, eid)
                expected_h = int(sample_event["timestep"].max()) + 1
                if expected_h != horizon:
                    raise ValueError(
                        f"Sample horizon mismatch for model={model_id} event={eid}: "
                        f"sample={expected_h}, data={horizon}"
                    )
                rows = map_event_predictions_to_rows(sample_event, pred1, pred2)
                table = pa.Table.from_pandas(rows, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(str(output_prediction_rows), table.schema, compression="snappy")
                writer.write_table(table)
                total_rows += len(rows)

                print(
                    f"[model {model_id}] {idx}/{len(test_events)} event_{eid}: "
                    f"rows={len(rows):,}, "
                    f"pred1d=({pred1.min():.3f},{pred1.max():.3f}), "
                    f"pred2d=({pred2.min():.3f},{pred2.max():.3f})"
                )
    finally:
        if writer is not None:
            writer.close()
    return total_rows


def assemble_submission(sample_submission: Path, prediction_rows: Path, output_submission: Path) -> None:
    con = duckdb.connect(database=":memory:")
    try:
        sample_q = str(sample_submission).replace("'", "''")
        pred_q = str(prediction_rows).replace("'", "''")
        out_q = str(output_submission).replace("'", "''")
        missing = con.execute(
            """
            SELECT COUNT(*)::BIGINT
            FROM read_parquet(?) s
            LEFT JOIN read_parquet(?) p USING (row_id)
            WHERE p.water_level IS NULL
            """,
            [str(sample_submission), str(prediction_rows)],
        ).fetchone()[0]
        if int(missing) != 0:
            raise ValueError(f"Missing predictions for {missing} rows")

        output_submission.parent.mkdir(parents=True, exist_ok=True)
        con.execute(
            f"""
            COPY (
                SELECT
                    s.row_id,
                    s.model_id,
                    s.event_id,
                    s.node_type,
                    s.node_id,
                    p.water_level
                FROM read_parquet('{sample_q}') s
                JOIN read_parquet('{pred_q}') p USING (row_id)
                ORDER BY s.row_id
            ) TO '{out_q}' (FORMAT PARQUET, COMPRESSION SNAPPY)
            """
        )
    finally:
        con.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit ARX baselines and create Kaggle submission parquet.")
    parser.add_argument("--data_dir", type=str, default="./data", help="Competition data root")
    parser.add_argument("--sample_submission", type=str, default="data/sample_submission.parquet", help="Sample parquet")
    parser.add_argument("--prefix_len", type=int, default=10, help="Observed warmup length")
    parser.add_argument("--val_train_fraction", type=float, default=0.8, help="Train fraction for holdout validation")
    parser.add_argument("--skip_validation", action="store_true", help="Skip holdout evaluation")
    parser.add_argument(
        "--model2_use_retrieval_blend",
        action="store_true",
        help="Blend Model 2 nodewise ARX with rainfall-retrieval shifted trajectories",
    )
    parser.add_argument(
        "--model2_arx_weight",
        type=float,
        default=0.5,
        help="Fallback ARX weight for Model 2 blend if per-domain weights are not provided",
    )
    parser.add_argument(
        "--model2_arx_weight_1d",
        type=float,
        default=None,
        help="Model 2 ARX weight for 1D (0=retrieval only, 1=ARX only)",
    )
    parser.add_argument(
        "--model2_arx_weight_2d",
        type=float,
        default=None,
        help="Model 2 ARX weight for 2D (0=retrieval only, 1=ARX only)",
    )
    parser.add_argument(
        "--model2_retrieval_top_k",
        type=int,
        default=1,
        help="Top-k neighbors for retrieval blend (inverse-distance weighted)",
    )
    parser.add_argument(
        "--model2_auto_tune_blend",
        action="store_true",
        help="Tune Model 2 ARX blend weight on holdout split",
    )
    parser.add_argument(
        "--model2_blend_grid",
        type=str,
        default="0.3,0.5,0.7",
        help="Fallback 1D/2D blend grid if per-domain grids are not provided",
    )
    parser.add_argument(
        "--model2_blend_grid_1d",
        type=str,
        default="",
        help="Comma-separated 1D ARX weights for auto-tuning",
    )
    parser.add_argument(
        "--model2_blend_grid_2d",
        type=str,
        default="",
        help="Comma-separated 2D ARX weights for auto-tuning",
    )
    parser.add_argument(
        "--output_prediction_rows",
        type=str,
        default="pred_rows_arx_all_models.parquet",
        help="Output row-wise predictions parquet (row_id, water_level)",
    )
    parser.add_argument(
        "--output_submission",
        type=str,
        default="submission_arx_all_models.parquet",
        help="Final submission parquet path",
    )
    parser.add_argument("--metrics_json", type=str, default="", help="Optional path to save holdout metrics JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    sample_submission = Path(args.sample_submission).resolve()
    out_rows = Path(args.output_prediction_rows).resolve()
    out_sub = Path(args.output_submission).resolve()
    metrics_path = Path(args.metrics_json).resolve() if args.metrics_json else None
    model2_w1 = float(args.model2_arx_weight if args.model2_arx_weight_1d is None else args.model2_arx_weight_1d)
    model2_w2 = float(args.model2_arx_weight if args.model2_arx_weight_2d is None else args.model2_arx_weight_2d)

    print("=" * 70)
    print("ARX Submission Pipeline")
    print(f"Data dir: {data_dir}")
    print(f"Sample submission: {sample_submission}")
    print(f"Prefix len: {args.prefix_len}")
    if args.model2_use_retrieval_blend:
        print(
            "Model 2 mode: blended "
            f"(arx_weight_1d={model2_w1:.3f}, arx_weight_2d={model2_w2:.3f}, "
            f"top_k={args.model2_retrieval_top_k})"
        )
    else:
        print("Model 2 mode: node-wise ARX only")
    print("=" * 70)

    metrics: Dict[str, Dict[str, float]] = {}

    # Model 1
    m1_events = list_events(data_dir, 1, "train")
    m1_train, m1_val = split_events(m1_events, args.val_train_fraction)
    m1_std_1d, m1_std_2d = compute_target_stds(data_dir, 1, m1_train)
    print(f"Model 1: train_events={len(m1_train)}, val_events={len(m1_val)}")

    if not args.skip_validation:
        m1_eval_model = fit_shared_arx(data_dir, 1, m1_train)
        m1_metrics = evaluate_arx_model(
            data_dir=data_dir,
            model_id=1,
            model=m1_eval_model,
            val_events=m1_val,
            prefix_len=args.prefix_len,
            std_1d=m1_std_1d,
            std_2d=m1_std_2d,
        )
        metrics["model_1"] = m1_metrics
        print(f"Model 1 holdout: {json.dumps(m1_metrics, indent=2)}")

    m1_final = fit_shared_arx(data_dir, 1, m1_events)
    print("Model 1 final fit complete (shared ARX).")

    # Model 2
    m2_events = list_events(data_dir, 2, "train")
    m2_train, m2_val = split_events(m2_events, args.val_train_fraction)
    m2_std_1d, m2_std_2d = compute_target_stds(data_dir, 2, m2_train)
    print(f"Model 2: train_events={len(m2_train)}, val_events={len(m2_val)}")

    best_blend_weight_1d = model2_w1
    best_blend_weight_2d = model2_w2

    if not args.skip_validation:
        m2_eval_arx = fit_nodewise_arx_model2(data_dir, m2_train)
        m2_metrics_arx = evaluate_arx_model(
            data_dir=data_dir,
            model_id=2,
            model=m2_eval_arx,
            val_events=m2_val,
            prefix_len=args.prefix_len,
            std_1d=m2_std_1d,
            std_2d=m2_std_2d,
        )
        metrics["model_2_arx_only"] = m2_metrics_arx
        print(f"Model 2 holdout (arx only): {json.dumps(m2_metrics_arx, indent=2)}")

        if args.model2_use_retrieval_blend:
            m2_eval_bank = build_model2_retrieval_bank(data_dir, m2_train, prefix_len=args.prefix_len)
            if args.model2_auto_tune_blend:
                candidates = [
                    float(x.strip()) for x in args.model2_blend_grid.split(",") if x.strip()
                ]
                if not candidates:
                    raise ValueError("No valid blend weights in --model2_blend_grid")
                grid_results: Dict[str, Dict[str, float]] = {}
                best_score = float("inf")
                grid_1d = (
                    [float(x.strip()) for x in args.model2_blend_grid_1d.split(",") if x.strip()]
                    if args.model2_blend_grid_1d.strip()
                    else candidates
                )
                grid_2d = (
                    [float(x.strip()) for x in args.model2_blend_grid_2d.split(",") if x.strip()]
                    if args.model2_blend_grid_2d.strip()
                    else candidates
                )
                for w1 in grid_1d:
                    for w2 in grid_2d:
                        model_blend = BlendedModel2(
                            arx=m2_eval_arx,
                            retrieval_bank=m2_eval_bank,
                            arx_weight_1d=w1,
                            arx_weight_2d=w2,
                            retrieval_top_k=args.model2_retrieval_top_k,
                        )
                        m = evaluate_arx_model(
                            data_dir=data_dir,
                            model_id=2,
                            model=model_blend,
                            val_events=m2_val,
                            prefix_len=args.prefix_len,
                            std_1d=m2_std_1d,
                            std_2d=m2_std_2d,
                        )
                        grid_results[f"arx_weight_1d_{w1:.4f}_2d_{w2:.4f}"] = m
                        if m["std_rmse"] < best_score:
                            best_score = m["std_rmse"]
                            best_blend_weight_1d = w1
                            best_blend_weight_2d = w2
                metrics["model_2_blend_grid"] = grid_results
                print(f"Model 2 blend grid: {json.dumps(grid_results, indent=2)}")
                print(
                    "Model 2 selected blend: "
                    f"arx_weight_1d={best_blend_weight_1d:.4f}, "
                    f"arx_weight_2d={best_blend_weight_2d:.4f}"
                )

            m2_eval_blend = BlendedModel2(
                arx=m2_eval_arx,
                retrieval_bank=m2_eval_bank,
                arx_weight_1d=best_blend_weight_1d,
                arx_weight_2d=best_blend_weight_2d,
                retrieval_top_k=args.model2_retrieval_top_k,
            )
            m2_metrics_blend = evaluate_arx_model(
                data_dir=data_dir,
                model_id=2,
                model=m2_eval_blend,
                val_events=m2_val,
                prefix_len=args.prefix_len,
                std_1d=m2_std_1d,
                std_2d=m2_std_2d,
            )
            metrics["model_2_blended"] = m2_metrics_blend
            print(f"Model 2 holdout (blended): {json.dumps(m2_metrics_blend, indent=2)}")

    m2_final_arx = fit_nodewise_arx_model2(data_dir, m2_events)
    if args.model2_use_retrieval_blend:
        m2_final_bank = build_model2_retrieval_bank(data_dir, m2_events, prefix_len=args.prefix_len)
        m2_final = BlendedModel2(
            arx=m2_final_arx,
            retrieval_bank=m2_final_bank,
            arx_weight_1d=best_blend_weight_1d,
            arx_weight_2d=best_blend_weight_2d,
            retrieval_top_k=args.model2_retrieval_top_k,
        )
        print(
            "Model 2 final fit complete (blended): "
            f"arx_weight_1d={best_blend_weight_1d:.4f}, "
            f"arx_weight_2d={best_blend_weight_2d:.4f}, "
            f"top_k={args.model2_retrieval_top_k}"
        )
    else:
        m2_final = m2_final_arx
        print("Model 2 final fit complete (node-wise ARX).")

    if metrics_path is not None and metrics:
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(metrics, indent=2))
        print(f"Saved holdout metrics: {metrics_path}")

    total_rows = write_prediction_rows(
        data_dir=data_dir,
        sample_submission=sample_submission,
        model_1=m1_final,
        model_2=m2_final,
        prefix_len=args.prefix_len,
        output_prediction_rows=out_rows,
    )
    print(f"Saved row-wise predictions: {out_rows} ({total_rows:,} rows)")

    assemble_submission(
        sample_submission=sample_submission,
        prediction_rows=out_rows,
        output_submission=out_sub,
    )
    print(f"Saved final submission: {out_sub}")

    # Basic sanity report
    df = pd.read_parquet(out_sub)
    print(f"Rows: {len(df):,}")
    print(f"Null water_level: {int(df['water_level'].isna().sum())}")
    print(
        f"Model 1 range: ({df[df.model_id == 1]['water_level'].min():.3f}, "
        f"{df[df.model_id == 1]['water_level'].max():.3f})"
    )
    print(
        f"Model 2 range: ({df[df.model_id == 2]['water_level'].min():.3f}, "
        f"{df[df.model_id == 2]['water_level'].max():.3f})"
    )


if __name__ == "__main__":
    main()
