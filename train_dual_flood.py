#!/usr/bin/env python3
"""
DualFlood v2: Latent-Space Flood Prediction with Supervised Edge Flows

Combines the best of both approaches:
- VGSSM's powerful latent space (c_e event latent, z_t temporal latent)
- DualFlood's supervised edge flow prediction

Architecture:
1. Prefix Encoder (GRU): Encode full prefix sequence into latent
2. Event Latent c_e: Captures storm/event characteristics (inferred from prefix)
3. Temporal Latent z_t: Evolves via transition model during rollout
4. Decoder: Maps z_t -> water levels AND edge flows
5. Supervised loss on BOTH node and edge predictions

Key insight: The latent space provides powerful temporal modeling,
while supervised edge flows provide physical meaning.
"""

import os
import math
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
from typing import Dict, List, Tuple, Optional
from pathlib import Path

STATIC_1D_COLS = [
    'position_x', 'position_y', 'depth', 'invert_elevation',
    'surface_elevation', 'base_area'
]
STATIC_2D_COLS = [
    'position_x', 'position_y', 'area', 'roughness', 'min_elevation',
    'elevation', 'aspect', 'curvature', 'flow_accumulation'
]
EDGE_1D_COLS = ['length', 'diameter', 'roughness', 'slope']
EDGE_2D_CANDIDATE_COLS = ['length', 'width', 'depth', 'roughness']


def to_numpy_float32(x) -> np.ndarray:
    """Convert tensors/lists/scalars to float32 numpy arrays."""
    if isinstance(x, np.ndarray):
        return x.astype(np.float32)
    if torch.is_tensor(x):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def load_matching_state_dict(module: nn.Module, candidate_state: Dict[str, torch.Tensor]) -> Tuple[List[str], List[str]]:
    """
    Load only checkpoint parameters with matching names AND shapes.
    Returns:
        skipped_keys: present in checkpoint but not loadable (missing or shape mismatch)
        missing_keys: present in module but absent after filtered load
    """
    module_state = module.state_dict()
    filtered = {}
    skipped_keys = []
    for key, value in candidate_state.items():
        if key not in module_state:
            skipped_keys.append(key)
            continue
        if tuple(module_state[key].shape) != tuple(value.shape):
            skipped_keys.append(key)
            continue
        filtered[key] = value

    module.load_state_dict(filtered, strict=False)
    missing_keys = [k for k in module_state.keys() if k not in filtered]
    return skipped_keys, missing_keys


def adapt_init_state_for_new_architecture(
    model_state: Dict[str, torch.Tensor],
    use_moe_transition: bool = False,
    moe_num_experts: int = 4,
    use_dual_timescale_latent: bool = False,
) -> Dict[str, torch.Tensor]:
    """
    Adapt older checkpoint weights to newer architectural variants.

    - Seeds MoE experts from dense local_mlp weights when available.
    - Seeds slow transitions from fast transitions for dual-timescale models.
    """
    adapted = dict(model_state)

    if use_moe_transition:
        for prefix in ("transition_1d", "transition_2d"):
            dense_prefix = f"{prefix}.local_mlp."
            expert_prefix = f"{prefix}.local_experts."
            has_dense = any(k.startswith(dense_prefix) for k in adapted.keys())
            has_expert = any(k.startswith(expert_prefix) for k in adapted.keys())
            if not has_dense or has_expert:
                continue

            dense_items = [(k, v) for k, v in adapted.items() if k.startswith(dense_prefix)]
            for i in range(max(1, int(moe_num_experts))):
                for key, value in dense_items:
                    new_key = key.replace(dense_prefix, f"{expert_prefix}{i}.")
                    if new_key not in adapted:
                        adapted[new_key] = value.clone()

    if use_dual_timescale_latent:
        for fast_prefix, slow_prefix in (
            ("transition_1d.", "transition_1d_slow."),
            ("transition_2d.", "transition_2d_slow."),
        ):
            for key, value in list(adapted.items()):
                if not key.startswith(fast_prefix):
                    continue
                slow_key = slow_prefix + key[len(fast_prefix):]
                if slow_key not in adapted:
                    adapted[slow_key] = value.clone()

    return adapted


# ==============================================================================
# DATASET WITH EDGE FLOW DATA
# ==============================================================================

class DualFloodDataset(Dataset):
    """
    Dataset that loads BOTH node dynamics AND edge dynamics (flows).

    This is the key fix - we now have ground truth flows for supervised training.
    """

    def __init__(
        self,
        data_dir: str,
        model_id: int,
        event_id: int,
        split: str,
        seq_len: int = 10,
        pred_len: int = 90,
        stride: int = 1,
        normalize: bool = True,
        normalization_stats: Optional[Dict] = None,
        min_pred_len: int = 1,
        future_inlet_mode: str = "observed",
        start_only: bool = False,
        future_inlet_dropout_prob: float = 0.0,
        future_inlet_seq_dropout_prob: float = 0.0,
    ):
        self.data_dir = data_dir
        self.model_id = model_id
        self.event_id = event_id
        self.split = split
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stride = stride
        self.normalize = normalize
        self.min_pred_len = max(1, int(min_pred_len))
        self.future_inlet_mode = future_inlet_mode
        self.start_only = start_only
        self.future_inlet_dropout_prob = future_inlet_dropout_prob
        self.future_inlet_seq_dropout_prob = future_inlet_seq_dropout_prob

        if self.min_pred_len > self.pred_len:
            raise ValueError(
                f"min_pred_len={self.min_pred_len} cannot exceed pred_len={self.pred_len}"
            )

        valid_inlet_modes = {"observed", "missing", "mixed", "zero", "last"}
        if self.future_inlet_mode not in valid_inlet_modes:
            raise ValueError(
                f"Invalid future_inlet_mode='{self.future_inlet_mode}'. "
                f"Expected one of {sorted(valid_inlet_modes)}."
            )

        if not (0.0 <= self.future_inlet_dropout_prob <= 1.0):
            raise ValueError("future_inlet_dropout_prob must be in [0, 1]")
        if not (0.0 <= self.future_inlet_seq_dropout_prob <= 1.0):
            raise ValueError("future_inlet_seq_dropout_prob must be in [0, 1]")

        # Load all data
        self._load_data()

        # Compute or use provided normalization stats
        if normalization_stats is not None:
            self.norm_stats = normalization_stats
        elif normalize:
            self.norm_stats = self._compute_normalization_stats()
        else:
            self.norm_stats = None

        self._prepare_normalization_constants()

        # Build sequences
        self._build_sequences()

    def _load_data(self):
        """Load node AND edge dynamic data."""
        event_path = os.path.join(
            self.data_dir, f"Model_{self.model_id}", self.split, f"event_{self.event_id}"
        )

        # Load node dynamics
        df_1d_nodes = pd.read_csv(os.path.join(event_path, "1d_nodes_dynamic_all.csv"))
        df_2d_nodes = pd.read_csv(os.path.join(event_path, "2d_nodes_dynamic_all.csv"))

        # Load edge dynamics (THE KEY ADDITION!)
        df_1d_edges = pd.read_csv(os.path.join(event_path, "1d_edges_dynamic_all.csv"))
        df_2d_edges = pd.read_csv(os.path.join(event_path, "2d_edges_dynamic_all.csv"))

        # Get dimensions
        self.num_timesteps = df_1d_nodes['timestep'].nunique()
        self.num_1d_nodes = df_1d_nodes['node_idx'].nunique()
        self.num_2d_nodes = df_2d_nodes['node_idx'].nunique()
        self.num_1d_edges = df_1d_edges['edge_idx'].nunique()
        self.num_2d_edges = df_2d_edges['edge_idx'].nunique()

        # Reshape node data to [T, N, F]
        self.node_1d = self._reshape_node_data(
            df_1d_nodes, self.num_1d_nodes,
            ['water_level', 'inlet_flow']
        )
        self.node_2d = self._reshape_node_data(
            df_2d_nodes, self.num_2d_nodes,
            ['water_level', 'rainfall', 'water_volume']
        )

        # Reshape edge data to [T, E, F]
        self.edge_1d = self._reshape_edge_data(
            df_1d_edges, self.num_1d_edges,
            ['flow', 'velocity']
        )
        self.edge_2d = self._reshape_edge_data(
            df_2d_edges, self.num_2d_edges,
            ['flow', 'velocity']
        )

        # Extract specific variables for convenience
        self.water_level_1d = self.node_1d[:, :, 0:1]  # [T, N, 1]
        self.water_level_2d = self.node_2d[:, :, 0:1]  # [T, N, 1]
        self.flow_1d = self.edge_1d[:, :, 0:1]  # [T, E, 1]
        self.flow_2d = self.edge_2d[:, :, 0:1]  # [T, E, 1]
        self.rainfall = self.node_2d[:, :, 1:2]  # [T, N, 1]
        self.inlet_flow = self.node_1d[:, :, 1:2]  # [T, N, 1]

    def _reshape_node_data(self, df: pd.DataFrame, num_nodes: int, vars: List[str]) -> np.ndarray:
        """Reshape node dataframe to [T, N, F] array."""
        available_vars = [v for v in vars if v in df.columns]
        if not available_vars:
            return np.zeros((self.num_timesteps, num_nodes, 1), dtype=np.float32)

        df = df.sort_values(['timestep', 'node_idx'])
        num_timesteps = df['timestep'].nunique()

        data = df[available_vars].values.astype(np.float32)
        data = data.reshape(num_timesteps, num_nodes, len(available_vars))

        # Handle NaN
        data = np.nan_to_num(data, nan=0.0)
        return data

    def _reshape_edge_data(self, df: pd.DataFrame, num_edges: int, vars: List[str]) -> np.ndarray:
        """Reshape edge dataframe to [T, E, F] array."""
        available_vars = [v for v in vars if v in df.columns]
        if not available_vars:
            return np.zeros((self.num_timesteps, num_edges, 1), dtype=np.float32)

        df = df.sort_values(['timestep', 'edge_idx'])
        num_timesteps = df['timestep'].nunique()

        data = df[available_vars].values.astype(np.float32)
        data = data.reshape(num_timesteps, num_edges, len(available_vars))

        # Handle NaN
        data = np.nan_to_num(data, nan=0.0)
        return data

    def _compute_normalization_stats(self) -> Dict:
        """Compute normalization statistics."""
        return {
            'node_1d': {
                'mean': self.node_1d.mean(axis=(0, 1)),
                'std': self.node_1d.std(axis=(0, 1)) + 1e-8
            },
            'node_2d': {
                'mean': self.node_2d.mean(axis=(0, 1)),
                'std': self.node_2d.std(axis=(0, 1)) + 1e-8
            },
            'water_level_1d': {
                'mean': self.water_level_1d.mean(),
                'std': self.water_level_1d.std() + 1e-8
            },
            'water_level_2d': {
                'mean': self.water_level_2d.mean(),
                'std': self.water_level_2d.std() + 1e-8
            },
            'flow_1d': {
                'mean': self.flow_1d.mean(),
                'std': self.flow_1d.std() + 1e-8
            },
            'flow_2d': {
                'mean': self.flow_2d.mean(),
                'std': self.flow_2d.std() + 1e-8
            },
        }

    def _prepare_normalization_constants(self) -> None:
        if self.norm_stats is None:
            self.node_1d_mean = None
            self.node_1d_std = None
            self.node_2d_mean = None
            self.node_2d_std = None
            self.wl_1d_mean = 0.0
            self.wl_1d_std = 1.0
            self.wl_2d_mean = 0.0
            self.wl_2d_std = 1.0
            self.flow_1d_mean = 0.0
            self.flow_1d_std = 1.0
            self.flow_2d_mean = 0.0
            self.flow_2d_std = 1.0
            self.rain_mean = 0.0
            self.rain_std = 1.0
            self.inlet_mean = 0.0
            self.inlet_std = 1.0
            return

        self.node_1d_mean = to_numpy_float32(self.norm_stats['node_1d']['mean']).reshape(1, 1, -1)
        self.node_1d_std = to_numpy_float32(self.norm_stats['node_1d']['std']).reshape(1, 1, -1)
        self.node_2d_mean = to_numpy_float32(self.norm_stats['node_2d']['mean']).reshape(1, 1, -1)
        self.node_2d_std = to_numpy_float32(self.norm_stats['node_2d']['std']).reshape(1, 1, -1)
        self.wl_1d_mean = float(self.norm_stats['water_level_1d']['mean'])
        self.wl_1d_std = float(self.norm_stats['water_level_1d']['std'])
        self.wl_2d_mean = float(self.norm_stats['water_level_2d']['mean'])
        self.wl_2d_std = float(self.norm_stats['water_level_2d']['std'])
        self.flow_1d_mean = float(self.norm_stats['flow_1d']['mean'])
        self.flow_1d_std = float(self.norm_stats['flow_1d']['std'])
        self.flow_2d_mean = float(self.norm_stats['flow_2d']['mean'])
        self.flow_2d_std = float(self.norm_stats['flow_2d']['std'])
        self.rain_mean = float(self.norm_stats['node_2d']['mean'][1])
        self.rain_std = float(self.norm_stats['node_2d']['std'][1])
        self.inlet_mean = float(self.norm_stats['node_1d']['mean'][1])
        self.inlet_std = float(self.norm_stats['node_1d']['std'][1])

    def _build_sequences(self):
        """Build list of valid sequence start indices."""
        min_total_len = self.seq_len + self.min_pred_len
        self.valid_starts = []
        self.valid_horizons = []

        if self.start_only:
            if self.num_timesteps >= min_total_len:
                self.valid_starts.append(0)
                self.valid_horizons.append(min(self.pred_len, self.num_timesteps - self.seq_len))
            return

        max_start = self.num_timesteps - min_total_len
        for start in range(0, max_start + 1, self.stride):
            end_input = start + self.seq_len
            horizon = min(self.pred_len, self.num_timesteps - end_input)
            self.valid_starts.append(start)
            self.valid_horizons.append(horizon)

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start = self.valid_starts[idx]
        horizon = self.valid_horizons[idx]
        end_input = start + self.seq_len
        if horizon < self.min_pred_len:
            raise IndexError(
                f"Sequence at idx={idx} has horizon={horizon}, below min_pred_len={self.min_pred_len}"
            )
        end_target = end_input + horizon

        def _pad_time(arr: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
            if arr.shape[0] == self.pred_len:
                return arr
            pad_shape = (self.pred_len - arr.shape[0],) + arr.shape[1:]
            pad = np.full(pad_shape, fill_value, dtype=np.float32)
            return np.concatenate([arr, pad], axis=0)

        # Input sequences (prefix)
        input_1d = self.node_1d[start:end_input].astype(np.float32, copy=False)
        input_2d = self.node_2d[start:end_input].astype(np.float32, copy=False)
        if self.norm_stats is not None:
            input_1d = (input_1d - self.node_1d_mean) / self.node_1d_std
            input_2d = (input_2d - self.node_2d_mean) / self.node_2d_std

        # Target sequences (future)
        target_wl_1d = self.water_level_1d[end_input:end_target].astype(np.float32, copy=False)
        target_wl_2d = self.water_level_2d[end_input:end_target].astype(np.float32, copy=False)
        if self.norm_stats is not None:
            target_wl_1d = (target_wl_1d - self.wl_1d_mean) / self.wl_1d_std
            target_wl_2d = (target_wl_2d - self.wl_2d_mean) / self.wl_2d_std
        target_wl_1d = _pad_time(target_wl_1d, fill_value=0.0)
        target_wl_2d = _pad_time(target_wl_2d, fill_value=0.0)

        # Target flows (THE KEY ADDITION!)
        target_flow_1d = self.flow_1d[end_input:end_target].astype(np.float32, copy=False)
        target_flow_2d = self.flow_2d[end_input:end_target].astype(np.float32, copy=False)
        if self.norm_stats is not None:
            target_flow_1d = (target_flow_1d - self.flow_1d_mean) / self.flow_1d_std
            target_flow_2d = (target_flow_2d - self.flow_2d_mean) / self.flow_2d_std
        target_flow_1d = _pad_time(target_flow_1d, fill_value=0.0)
        target_flow_2d = _pad_time(target_flow_2d, fill_value=0.0)

        # Future covariates (rainfall, inlet flow)
        future_rainfall = self.rainfall[end_input:end_target].astype(np.float32, copy=False)
        future_inlet = self.inlet_flow[end_input:end_target].astype(np.float32, copy=False)
        future_inlet_mask = np.ones_like(future_inlet, dtype=np.float32)

        # Test data has unknown future inlet after prefix. Allow robust modes.
        if self.future_inlet_mode == 'zero':
            future_inlet = np.zeros_like(future_inlet, dtype=np.float32)
        elif self.future_inlet_mode == 'last':
            last_obs_inlet = self.inlet_flow[end_input - 1:end_input].astype(np.float32, copy=False)
            future_inlet = np.repeat(last_obs_inlet, future_inlet.shape[0], axis=0).astype(np.float32, copy=False)

        if self.norm_stats is not None:
            future_rainfall = (future_rainfall - self.rain_mean) / self.rain_std
            future_inlet = (future_inlet - self.inlet_mean) / self.inlet_std

        # Missing inlet is represented with mean-imputed normalized value (0) + mask.
        if self.future_inlet_mode == 'missing':
            future_inlet = np.zeros_like(future_inlet, dtype=np.float32)
            future_inlet_mask = np.zeros_like(future_inlet_mask, dtype=np.float32)
        elif self.future_inlet_mode == 'mixed':
            drop_mask = np.random.random(size=future_inlet.shape) < self.future_inlet_dropout_prob
            if np.random.random() < self.future_inlet_seq_dropout_prob:
                drop_mask[:] = True
            future_inlet = future_inlet.copy()
            future_inlet[drop_mask] = 0.0
            future_inlet_mask[drop_mask] = 0.0
        elif self.future_inlet_mode == 'zero':
            # Zero is an observed physical value (not missing).
            future_inlet_mask = np.ones_like(future_inlet_mask, dtype=np.float32)
        elif self.future_inlet_mode == 'last':
            # Last-value carry-forward is observed/imputed; keep mask=1 for compatibility.
            future_inlet_mask = np.ones_like(future_inlet_mask, dtype=np.float32)

        future_rainfall = _pad_time(future_rainfall, fill_value=0.0)
        future_inlet = _pad_time(future_inlet, fill_value=0.0)
        future_inlet_mask = _pad_time(future_inlet_mask, fill_value=0.0)
        target_mask = np.zeros((self.pred_len, 1, 1), dtype=np.float32)
        target_mask[:horizon] = 1.0

        return {
            'input_1d': torch.from_numpy(input_1d),
            'input_2d': torch.from_numpy(input_2d),
            'target_wl_1d': torch.from_numpy(target_wl_1d),
            'target_wl_2d': torch.from_numpy(target_wl_2d),
            'target_flow_1d': torch.from_numpy(target_flow_1d),
            'target_flow_2d': torch.from_numpy(target_flow_2d),
            'future_rainfall': torch.from_numpy(future_rainfall),
            'future_inlet': torch.from_numpy(future_inlet),
            'future_inlet_mask': torch.from_numpy(future_inlet_mask),
            'target_mask': torch.from_numpy(target_mask),
            'target_len': torch.tensor(horizon, dtype=torch.long),
            'prefix_len': self.seq_len,
        }


class DualFloodDataModule(pl.LightningDataModule):
    """DataModule for DualFlood training."""

    def __init__(
        self,
        data_dir: str,
        model_id: int,
        batch_size: int = 4,
        seq_len: int = 10,
        pred_len: int = 90,
        min_pred_len: int = 1,
        num_workers: int = 0,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        drop_last_train: bool = False,
        use_norm_cache: bool = True,
        norm_cache_path: str = "",
        future_inlet_mode_train: str = "observed",
        future_inlet_mode_val: str = "observed",
        train_start_only: bool = False,
        val_start_only: bool = False,
        future_inlet_dropout_prob_train: float = 0.0,
        future_inlet_seq_dropout_prob_train: float = 0.0,
        future_inlet_dropout_prob_val: float = 0.0,
        future_inlet_seq_dropout_prob_val: float = 0.0,
        horizon_sampling_power: float = 0.0,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.model_id = model_id
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.min_pred_len = max(1, int(min_pred_len))
        if self.min_pred_len > self.pred_len:
            raise ValueError("min_pred_len cannot exceed pred_len")
        self.num_workers = num_workers
        self.prefetch_factor = max(1, int(prefetch_factor))
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers
        self.drop_last_train = drop_last_train
        self.use_norm_cache = use_norm_cache
        self.norm_cache_path = norm_cache_path.strip()
        self.future_inlet_mode_train = future_inlet_mode_train
        self.future_inlet_mode_val = future_inlet_mode_val
        self.train_start_only = train_start_only
        self.val_start_only = val_start_only
        self.future_inlet_dropout_prob_train = future_inlet_dropout_prob_train
        self.future_inlet_seq_dropout_prob_train = future_inlet_seq_dropout_prob_train
        self.future_inlet_dropout_prob_val = future_inlet_dropout_prob_val
        self.future_inlet_seq_dropout_prob_val = future_inlet_seq_dropout_prob_val
        self.horizon_sampling_power = max(0.0, float(horizon_sampling_power))

        self.train_datasets = []
        self.val_datasets = []
        self.norm_stats = None
        self._is_setup = False

    @staticmethod
    def _init_running_stats() -> Dict[str, Optional[np.ndarray]]:
        return {"sum": None, "sumsq": None, "count": 0}

    @staticmethod
    def _update_running_stats(stats: Dict[str, Optional[np.ndarray]], arr: np.ndarray) -> None:
        flat = arr.astype(np.float64).reshape(-1, arr.shape[-1])
        arr_sum = flat.sum(axis=0)
        arr_sumsq = (flat ** 2).sum(axis=0)

        if stats["sum"] is None:
            stats["sum"] = arr_sum
            stats["sumsq"] = arr_sumsq
        else:
            stats["sum"] += arr_sum
            stats["sumsq"] += arr_sumsq

        stats["count"] += flat.shape[0]

    @staticmethod
    def _finalize_running_stats(stats: Dict[str, Optional[np.ndarray]]) -> Tuple[np.ndarray, np.ndarray]:
        mean = stats["sum"] / max(stats["count"], 1)
        var = stats["sumsq"] / max(stats["count"], 1) - mean ** 2
        std = np.sqrt(np.maximum(var, 1e-12))
        return mean.astype(np.float32), (std + 1e-8).astype(np.float32)

    def _compute_global_norm_stats(self, train_events: List[int]) -> Dict:
        """Compute normalization stats across ALL training events."""
        running = {
            "node_1d": self._init_running_stats(),
            "node_2d": self._init_running_stats(),
            "water_level_1d": self._init_running_stats(),
            "water_level_2d": self._init_running_stats(),
            "flow_1d": self._init_running_stats(),
            "flow_2d": self._init_running_stats(),
        }

        print(f"Computing global normalization from {len(train_events)} train events...")
        for i, eid in enumerate(train_events, 1):
            ds = DualFloodDataset(
                self.data_dir,
                self.model_id,
                eid,
                "train",
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                min_pred_len=self.min_pred_len,
                normalize=False,
            )

            self._update_running_stats(running["node_1d"], ds.node_1d)
            self._update_running_stats(running["node_2d"], ds.node_2d)
            self._update_running_stats(running["water_level_1d"], ds.water_level_1d)
            self._update_running_stats(running["water_level_2d"], ds.water_level_2d)
            self._update_running_stats(running["flow_1d"], ds.flow_1d)
            self._update_running_stats(running["flow_2d"], ds.flow_2d)

            if i % 10 == 0 or i == len(train_events):
                print(f"  Processed {i}/{len(train_events)} events")

        node_1d_mean, node_1d_std = self._finalize_running_stats(running["node_1d"])
        node_2d_mean, node_2d_std = self._finalize_running_stats(running["node_2d"])
        wl_1d_mean, wl_1d_std = self._finalize_running_stats(running["water_level_1d"])
        wl_2d_mean, wl_2d_std = self._finalize_running_stats(running["water_level_2d"])
        flow_1d_mean, flow_1d_std = self._finalize_running_stats(running["flow_1d"])
        flow_2d_mean, flow_2d_std = self._finalize_running_stats(running["flow_2d"])

        return {
            "node_1d": {"mean": node_1d_mean, "std": node_1d_std},
            "node_2d": {"mean": node_2d_mean, "std": node_2d_std},
            "water_level_1d": {"mean": float(wl_1d_mean[0]), "std": float(wl_1d_std[0])},
            "water_level_2d": {"mean": float(wl_2d_mean[0]), "std": float(wl_2d_std[0])},
            "flow_1d": {"mean": float(flow_1d_mean[0]), "std": float(flow_1d_std[0])},
            "flow_2d": {"mean": float(flow_2d_mean[0]), "std": float(flow_2d_std[0])},
        }

    def _default_norm_cache_path(self, train_events: List[int]) -> str:
        first_eid = train_events[0]
        last_eid = train_events[-1]
        train_root = os.path.join(self.data_dir, f"Model_{self.model_id}", "train")
        filename = (
            f"norm_stats_seq{self.seq_len}_pred{self.pred_len}_"
            f"events{len(train_events)}_{first_eid}-{last_eid}.npz"
        )
        return os.path.join(train_root, filename)

    def _load_norm_cache(self, cache_path: str, train_events: List[int]) -> Optional[Dict]:
        if not cache_path or not os.path.exists(cache_path):
            return None
        try:
            with np.load(cache_path) as z:
                expected = (
                    int(z["meta_model_id"]) == int(self.model_id)
                    and int(z["meta_seq_len"]) == int(self.seq_len)
                    and int(z["meta_pred_len"]) == int(self.pred_len)
                    and int(z["meta_num_train_events"]) == int(len(train_events))
                )
                if not expected:
                    return None

                return {
                    "node_1d": {
                        "mean": z["node_1d_mean"].astype(np.float32),
                        "std": z["node_1d_std"].astype(np.float32),
                    },
                    "node_2d": {
                        "mean": z["node_2d_mean"].astype(np.float32),
                        "std": z["node_2d_std"].astype(np.float32),
                    },
                    "water_level_1d": {
                        "mean": float(z["water_level_1d_mean"]),
                        "std": float(z["water_level_1d_std"]),
                    },
                    "water_level_2d": {
                        "mean": float(z["water_level_2d_mean"]),
                        "std": float(z["water_level_2d_std"]),
                    },
                    "flow_1d": {
                        "mean": float(z["flow_1d_mean"]),
                        "std": float(z["flow_1d_std"]),
                    },
                    "flow_2d": {
                        "mean": float(z["flow_2d_mean"]),
                        "std": float(z["flow_2d_std"]),
                    },
                }
        except Exception as exc:
            print(f"Warning: failed to load norm cache {cache_path}: {exc}")
            return None

    def _save_norm_cache(self, cache_path: str, norm_stats: Dict, train_events: List[int]) -> None:
        if not cache_path:
            return
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        try:
            np.savez_compressed(
                cache_path,
                node_1d_mean=to_numpy_float32(norm_stats["node_1d"]["mean"]),
                node_1d_std=to_numpy_float32(norm_stats["node_1d"]["std"]),
                node_2d_mean=to_numpy_float32(norm_stats["node_2d"]["mean"]),
                node_2d_std=to_numpy_float32(norm_stats["node_2d"]["std"]),
                water_level_1d_mean=np.float32(norm_stats["water_level_1d"]["mean"]),
                water_level_1d_std=np.float32(norm_stats["water_level_1d"]["std"]),
                water_level_2d_mean=np.float32(norm_stats["water_level_2d"]["mean"]),
                water_level_2d_std=np.float32(norm_stats["water_level_2d"]["std"]),
                flow_1d_mean=np.float32(norm_stats["flow_1d"]["mean"]),
                flow_1d_std=np.float32(norm_stats["flow_1d"]["std"]),
                flow_2d_mean=np.float32(norm_stats["flow_2d"]["mean"]),
                flow_2d_std=np.float32(norm_stats["flow_2d"]["std"]),
                meta_model_id=np.int64(self.model_id),
                meta_seq_len=np.int64(self.seq_len),
                meta_pred_len=np.int64(self.pred_len),
                meta_num_train_events=np.int64(len(train_events)),
            )
        except Exception as exc:
            print(f"Warning: failed to save norm cache {cache_path}: {exc}")

    def setup(self, stage=None):
        if self._is_setup:
            return

        # Find all events
        train_path = os.path.join(self.data_dir, f"Model_{self.model_id}", "train")
        event_dirs = [d for d in os.listdir(train_path) if d.startswith("event_")]
        event_ids = sorted([int(d.split("_")[1]) for d in event_dirs])

        # Split events: 80% train, 20% val
        n_train = int(len(event_ids) * 0.8)
        train_events = event_ids[:n_train]
        val_events = event_ids[n_train:]

        print(f"Model {self.model_id}: {len(train_events)} train, {len(val_events)} val events")

        # Load first event only to discover dimensions.
        first_ds = DualFloodDataset(
            self.data_dir, self.model_id, train_events[0], "train",
            seq_len=self.seq_len, pred_len=self.pred_len, min_pred_len=self.min_pred_len, normalize=False
        )

        # Compute normalization from all train events (not just the first event).
        cache_path = self.norm_cache_path or self._default_norm_cache_path(train_events)
        if self.use_norm_cache:
            cached = self._load_norm_cache(cache_path, train_events)
            if cached is not None:
                print(f"Loaded normalization stats from cache: {cache_path}")
                self.norm_stats = cached
            else:
                self.norm_stats = self._compute_global_norm_stats(train_events)
                self._save_norm_cache(cache_path, self.norm_stats, train_events)
                print(f"Saved normalization stats cache: {cache_path}")
        else:
            self.norm_stats = self._compute_global_norm_stats(train_events)

        # Store dimensions
        self.num_1d_nodes = first_ds.num_1d_nodes
        self.num_2d_nodes = first_ds.num_2d_nodes
        self.num_1d_edges = first_ds.num_1d_edges
        self.num_2d_edges = first_ds.num_2d_edges

        print(f"  1D nodes: {self.num_1d_nodes}, 2D nodes: {self.num_2d_nodes}")
        print(f"  1D edges: {self.num_1d_edges}, 2D edges: {self.num_2d_edges}")

        # Create train datasets
        self.train_datasets = []
        for eid in train_events:
            ds = DualFloodDataset(
                self.data_dir, self.model_id, eid, "train",
                seq_len=self.seq_len, pred_len=self.pred_len,
                min_pred_len=self.min_pred_len,
                normalize=True, normalization_stats=self.norm_stats,
                future_inlet_mode=self.future_inlet_mode_train,
                start_only=self.train_start_only,
                future_inlet_dropout_prob=self.future_inlet_dropout_prob_train,
                future_inlet_seq_dropout_prob=self.future_inlet_seq_dropout_prob_train,
            )
            self.train_datasets.append(ds)

        # Create val datasets
        self.val_datasets = []
        for eid in val_events:
            ds = DualFloodDataset(
                self.data_dir, self.model_id, eid, "train",
                seq_len=self.seq_len, pred_len=self.pred_len,
                min_pred_len=self.min_pred_len,
                normalize=True, normalization_stats=self.norm_stats,
                future_inlet_mode=self.future_inlet_mode_val,
                start_only=self.val_start_only,
                future_inlet_dropout_prob=self.future_inlet_dropout_prob_val,
                future_inlet_seq_dropout_prob=self.future_inlet_seq_dropout_prob_val,
            )
            self.val_datasets.append(ds)

        self._is_setup = True

    def train_dataloader(self):
        combined = torch.utils.data.ConcatDataset(self.train_datasets)
        loader_kwargs = {
            "batch_size": self.batch_size,
            "drop_last": self.drop_last_train,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        if self.horizon_sampling_power > 0.0:
            weight_chunks = []
            for ds in self.train_datasets:
                if len(ds.valid_horizons) == 0:
                    continue
                horizons = np.asarray(ds.valid_horizons, dtype=np.float64)
                denom = max(1, int(ds.pred_len))
                horizon_ratio = np.clip(horizons / float(denom), 1e-6, None)
                weights = np.power(horizon_ratio, self.horizon_sampling_power)
                weight_chunks.append(torch.from_numpy(weights).double())
            if weight_chunks:
                sample_weights = torch.cat(weight_chunks, dim=0)
                sampler = WeightedRandomSampler(
                    weights=sample_weights,
                    num_samples=int(sample_weights.numel()),
                    replacement=True,
                )
                loader_kwargs["sampler"] = sampler
                loader_kwargs["shuffle"] = False
            else:
                loader_kwargs["shuffle"] = True
        else:
            loader_kwargs["shuffle"] = True
        if self.num_workers > 0:
            if self.persistent_workers:
                loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(combined, **loader_kwargs)

    def val_dataloader(self):
        combined = torch.utils.data.ConcatDataset(self.val_datasets)
        loader_kwargs = {
            "batch_size": self.batch_size,
            "shuffle": False,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
        }
        if self.num_workers > 0:
            if self.persistent_workers:
                loader_kwargs["persistent_workers"] = True
            loader_kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(combined, **loader_kwargs)


# ==============================================================================
# GRAPH BUILDER
# ==============================================================================

class DualFloodGraphBuilder:
    """Build heterogeneous graph with edge indices for flow prediction."""

    def __init__(self, data_dir: str, model_id: int, static_norm_stats: Optional[Dict] = None):
        self.data_dir = data_dir
        self.model_id = model_id
        self.static_norm_stats = static_norm_stats

    def compute_static_norm_stats(self) -> Dict:
        """Compute static feature normalization stats from the training graph only."""
        train_path = os.path.join(self.data_dir, f"Model_{self.model_id}", "train")
        node_1d_static = pd.read_csv(os.path.join(train_path, "1d_nodes_static.csv"))
        node_2d_static = pd.read_csv(os.path.join(train_path, "2d_nodes_static.csv"))
        edge_1d_static = pd.read_csv(os.path.join(train_path, "1d_edges_static.csv"))
        edge_2d_static = pd.read_csv(os.path.join(train_path, "2d_edges_static.csv"))

        df_2d = node_2d_static.copy()
        df_2d['min_elevation'] = df_2d['min_elevation'].fillna(df_2d['elevation'])

        x_1d = node_1d_static[STATIC_1D_COLS].values.astype(np.float32)
        x_2d = df_2d[STATIC_2D_COLS].values.astype(np.float32)
        x_1d = np.nan_to_num(x_1d, nan=0.0)
        x_2d = np.nan_to_num(x_2d, nan=0.0)

        mean_1d = x_1d.mean(axis=0)
        std_1d = x_1d.std(axis=0) + 1e-8
        mean_2d = x_2d.mean(axis=0)
        std_2d = x_2d.std(axis=0) + 1e-8

        edge_1d_cols = [c for c in EDGE_1D_COLS if c in edge_1d_static.columns]
        edge_2d_cols = [c for c in EDGE_2D_CANDIDATE_COLS if c in edge_2d_static.columns]
        edge_1d = edge_1d_static[edge_1d_cols].values.astype(np.float32)
        edge_2d = edge_2d_static[edge_2d_cols].values.astype(np.float32) if edge_2d_cols else np.zeros((len(edge_2d_static), 1), dtype=np.float32)
        edge_1d = np.nan_to_num(edge_1d, nan=0.0)
        edge_2d = np.nan_to_num(edge_2d, nan=0.0)

        return {
            "node_1d": {"mean": mean_1d, "std": std_1d},
            "node_2d": {"mean": mean_2d, "std": std_2d},
            "edge_1d": {
                "cols": edge_1d_cols,
                "mean": edge_1d.mean(axis=0),
                "std": edge_1d.std(axis=0) + 1e-8,
            },
            "edge_2d": {
                "cols": edge_2d_cols if edge_2d_cols else ["dummy"],
                "mean": edge_2d.mean(axis=0),
                "std": edge_2d.std(axis=0) + 1e-8,
            },
        }

    def build(self, split: str = "train") -> HeteroData:
        base_path = os.path.join(self.data_dir, f"Model_{self.model_id}", split)

        # Load static node features
        node_1d_static = pd.read_csv(os.path.join(base_path, "1d_nodes_static.csv"))
        node_2d_static = pd.read_csv(os.path.join(base_path, "2d_nodes_static.csv"))

        # Load edge indices
        edge_1d_index = pd.read_csv(os.path.join(base_path, "1d_edge_index.csv"))
        edge_2d_index = pd.read_csv(os.path.join(base_path, "2d_edge_index.csv"))

        # Load edge static features
        edge_1d_static = pd.read_csv(os.path.join(base_path, "1d_edges_static.csv"))
        edge_2d_static = pd.read_csv(os.path.join(base_path, "2d_edges_static.csv"))

        # Load 1D-2D connections
        connections = pd.read_csv(os.path.join(base_path, "1d2d_connections.csv"))

        # Build HeteroData
        graph = HeteroData()

        # Node features
        x_1d = node_1d_static[STATIC_1D_COLS].values.astype(np.float32)
        x_1d = np.nan_to_num(x_1d, nan=0.0)
        if self.static_norm_stats is not None:
            mean_1d = to_numpy_float32(self.static_norm_stats["node_1d"]["mean"])
            std_1d = to_numpy_float32(self.static_norm_stats["node_1d"]["std"])
            x_1d = (x_1d - mean_1d.reshape(1, -1)) / std_1d.reshape(1, -1)
        graph['1d'].x = torch.from_numpy(x_1d)
        graph['1d'].num_nodes = len(x_1d)

        df_2d = node_2d_static.copy()
        df_2d['min_elevation'] = df_2d['min_elevation'].fillna(df_2d['elevation'])
        x_2d = df_2d[STATIC_2D_COLS].values.astype(np.float32)
        x_2d = np.nan_to_num(x_2d, nan=0.0)
        if self.static_norm_stats is not None:
            mean_2d = to_numpy_float32(self.static_norm_stats["node_2d"]["mean"])
            std_2d = to_numpy_float32(self.static_norm_stats["node_2d"]["std"])
            x_2d = (x_2d - mean_2d.reshape(1, -1)) / std_2d.reshape(1, -1)
        graph['2d'].x = torch.from_numpy(x_2d)
        graph['2d'].num_nodes = len(x_2d)

        # 1D pipe edges
        edge_index_1d_np = np.stack(
            [edge_1d_index['from_node'].values, edge_1d_index['to_node'].values],
            axis=0
        ).astype(np.int64)
        edge_index_1d = torch.from_numpy(edge_index_1d_np)
        graph['1d', 'pipe', '1d'].edge_index = edge_index_1d

        # 1D edge features
        edge_1d_cols = [c for c in EDGE_1D_COLS if c in edge_1d_static.columns]
        edge_attr_1d = edge_1d_static[edge_1d_cols].values.astype(np.float32)
        edge_attr_1d = np.nan_to_num(edge_attr_1d, nan=0.0)
        if self.static_norm_stats is not None and "edge_1d" in self.static_norm_stats:
            mean_e1 = to_numpy_float32(self.static_norm_stats["edge_1d"]["mean"])
            std_e1 = to_numpy_float32(self.static_norm_stats["edge_1d"]["std"])
            if mean_e1.shape[0] == edge_attr_1d.shape[1]:
                edge_attr_1d = (edge_attr_1d - mean_e1.reshape(1, -1)) / std_e1.reshape(1, -1)
        graph['1d', 'pipe', '1d'].edge_attr = torch.from_numpy(edge_attr_1d)

        # 2D surface edges
        edge_index_2d_np = np.stack(
            [edge_2d_index['from_node'].values, edge_2d_index['to_node'].values],
            axis=0
        ).astype(np.int64)
        edge_index_2d = torch.from_numpy(edge_index_2d_np)
        graph['2d', 'surface', '2d'].edge_index = edge_index_2d

        # 2D edge features
        actual_cols = [c for c in EDGE_2D_CANDIDATE_COLS if c in edge_2d_static.columns]
        if actual_cols:
            edge_attr_2d = edge_2d_static[actual_cols].values.astype(np.float32)
        else:
            edge_attr_2d = np.zeros((len(edge_2d_index), 1), dtype=np.float32)
        edge_attr_2d = np.nan_to_num(edge_attr_2d, nan=0.0)
        if self.static_norm_stats is not None and "edge_2d" in self.static_norm_stats:
            mean_e2 = to_numpy_float32(self.static_norm_stats["edge_2d"]["mean"])
            std_e2 = to_numpy_float32(self.static_norm_stats["edge_2d"]["std"])
            if mean_e2.shape[0] == edge_attr_2d.shape[1]:
                edge_attr_2d = (edge_attr_2d - mean_e2.reshape(1, -1)) / std_e2.reshape(1, -1)
        graph['2d', 'surface', '2d'].edge_attr = torch.from_numpy(edge_attr_2d)

        # 1D-2D coupling edges (bidirectional)
        edge_1d_to_2d_np = np.stack(
            [connections['node_1d'].values, connections['node_2d'].values],
            axis=0
        ).astype(np.int64)
        edge_1d_to_2d = torch.from_numpy(edge_1d_to_2d_np)
        graph['1d', 'couples_to', '2d'].edge_index = edge_1d_to_2d

        edge_2d_to_1d_np = np.stack(
            [connections['node_2d'].values, connections['node_1d'].values],
            axis=0
        ).astype(np.int64)
        edge_2d_to_1d = torch.from_numpy(edge_2d_to_1d_np)
        graph['2d', 'couples_from', '1d'].edge_index = edge_2d_to_1d

        return graph


# ==============================================================================
# LATENT SPACE COMPONENTS (from VGSSM)
# ==============================================================================

class PrefixEncoder(nn.Module):
    """
    GRU-based encoder for prefix sequence.
    Maps [B, T, N, F] -> [B, N, H] hidden representation.

    This is the KEY component that was missing in DualFlood v1.
    Instead of just using the last timestep, we encode the full prefix.
    """

    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Per-node GRU (shared across nodes)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, num_nodes, input_dim]
        Returns:
            h: [batch, num_nodes, hidden_dim]
        """
        B, T, N, F = x.shape

        # Project input
        x = self.input_proj(x)  # [B, T, N, H]

        # Reshape for per-node GRU: [B*N, T, H]
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, self.hidden_dim)

        # GRU encoding
        _, h_n = self.gru(x)  # h_n: [num_layers, B*N, H]

        # Take last layer's hidden state
        h = h_n[-1]  # [B*N, H]

        # Reshape back: [B, N, H]
        h = h.view(B, N, self.hidden_dim)

        return self.output_proj(h)


class EventLatentInference(nn.Module):
    """
    Infer event latent c_e from prefix encoding.

    c_e captures storm-level characteristics:
    - Total rainfall intensity
    - Spatial distribution of rainfall
    - Initial conditions

    Using VAE-style inference: q(c_e | prefix) = N(mu, sigma)
    """

    def __init__(self, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.latent_dim = latent_dim

        # Aggregate prefix encoding to single vector
        self.aggregate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # VAE heads
        self.mu_head = nn.Linear(hidden_dim, latent_dim)
        self.logvar_head = nn.Linear(hidden_dim, latent_dim)

    def forward(self, h_prefix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h_prefix: [batch, num_nodes, hidden_dim]
        Returns:
            c_e: [batch, latent_dim] - sampled event latent
            mu: [batch, latent_dim]
            logvar: [batch, latent_dim]
        """
        # Global aggregation (mean over nodes)
        h = h_prefix.mean(dim=1)  # [B, H]
        h = self.aggregate(h)

        # VAE parameters
        mu = self.mu_head(h)
        logvar = self.logvar_head(h).clamp(-10, 2)  # Stability

        # Reparameterization trick
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            c_e = mu + eps * std
        else:
            c_e = mu

        return c_e, mu, logvar


class TemporalLatentInference(nn.Module):
    """
    Infer initial temporal latent z_0 from prefix encoding.

    z_t captures per-timestep state:
    - Current water levels across network
    - Flow regime (subcritical, critical, supercritical)
    - Network state (flooded manholes, surcharged pipes)

    z_0 is the initial state that evolves via the transition model.
    """

    def __init__(self, hidden_dim: int, latent_dim: int, num_nodes: int):
        super().__init__()
        self.latent_dim = latent_dim

        # Per-node latent inference
        self.node_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim * 2),  # mu and logvar
        )

    def forward(self, h_prefix: torch.Tensor, c_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            h_prefix: [batch, num_nodes, hidden_dim]
            c_e: [batch, latent_dim] - event latent (for conditioning)
        Returns:
            z_0: [batch, num_nodes, latent_dim]
            mu: [batch, num_nodes, latent_dim]
            logvar: [batch, num_nodes, latent_dim]
        """
        B, N, H = h_prefix.shape

        # Condition on event latent
        c_e_expanded = c_e.unsqueeze(1).expand(-1, N, -1)  # [B, N, latent_dim]

        # We need to project c_e to match hidden_dim for concatenation
        # Actually, let's just use h_prefix and add c_e info via addition
        # after projection to latent_dim

        # Project to mu and logvar
        out = self.node_proj(h_prefix)  # [B, N, latent_dim*2]
        mu, logvar = out.chunk(2, dim=-1)
        logvar = logvar.clamp(-10, 2)

        # Add c_e influence to mu
        mu = mu + c_e_expanded * 0.1  # Small influence

        # Reparameterization
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z_0 = mu + eps * std
        else:
            z_0 = mu

        return z_0, mu, logvar


class MultiScale2DEncoder(nn.Module):
    """Inject coarse-scale 2D context back into node latents."""

    def __init__(self, latent_dim: int, hidden_dim: int, num_clusters: int = 128):
        super().__init__()
        self.requested_clusters = max(1, int(num_clusters))
        self.num_clusters = 1
        self.register_buffer("cluster_idx", torch.zeros(0, dtype=torch.long), persistent=False)
        self.register_buffer("coarse_edge_index", torch.zeros(2, 0, dtype=torch.long), persistent=False)

        self.coarse_mlp = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.node_proj = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.gate = nn.Linear(latent_dim * 2, latent_dim)

        # Start as near-identity so inherited checkpoints remain stable.
        nn.init.zeros_(self.coarse_mlp[-1].weight)
        nn.init.zeros_(self.coarse_mlp[-1].bias)
        nn.init.zeros_(self.node_proj[-1].weight)
        nn.init.zeros_(self.node_proj[-1].bias)

    @staticmethod
    def _build_clusters(pos_xy: torch.Tensor, requested_clusters: int) -> Tuple[torch.Tensor, int]:
        n = pos_xy.shape[0]
        if n <= 1 or requested_clusters <= 1:
            return torch.zeros(n, dtype=torch.long, device=pos_xy.device), 1

        side = max(2, int(math.sqrt(requested_clusters)))
        x = pos_xy[:, 0]
        y = pos_xy[:, 1]
        eps = 1e-6
        x_edges = torch.linspace(x.min(), x.max() + eps, side + 1, device=pos_xy.device)
        y_edges = torch.linspace(y.min(), y.max() + eps, side + 1, device=pos_xy.device)
        x_bin = torch.bucketize(x, x_edges[1:-1])
        y_bin = torch.bucketize(y, y_edges[1:-1])
        cluster_raw = x_bin * side + y_bin
        _, cluster_idx = torch.unique(cluster_raw, sorted=True, return_inverse=True)
        num_clusters = int(cluster_idx.max().item()) + 1
        return cluster_idx, max(1, num_clusters)

    def set_graph(self, pos_xy: torch.Tensor, edge_index: torch.Tensor) -> None:
        """Precompute coarse cluster assignment and coarse adjacency."""
        cluster_idx, num_clusters = self._build_clusters(pos_xy, self.requested_clusters)
        self.cluster_idx = cluster_idx.long()
        self.num_clusters = num_clusters

        src_cluster = cluster_idx[edge_index[0]]
        dst_cluster = cluster_idx[edge_index[1]]
        keep = src_cluster != dst_cluster

        if keep.any():
            coarse_edges = torch.stack([src_cluster[keep], dst_cluster[keep]], dim=0)
            coarse_edges = torch.unique(coarse_edges, dim=1)
        else:
            loops = torch.arange(num_clusters, device=cluster_idx.device, dtype=torch.long)
            coarse_edges = torch.stack([loops, loops], dim=0)

        self.coarse_edge_index = coarse_edges.long()

    def forward(self, z_2d: torch.Tensor) -> torch.Tensor:
        if self.cluster_idx.numel() == 0 or self.num_clusters <= 1:
            return z_2d

        B, N, L = z_2d.shape
        cluster_idx = self.cluster_idx.to(z_2d.device)
        if cluster_idx.shape[0] != N:
            return z_2d

        K = self.num_clusters
        node_idx = cluster_idx.view(1, N, 1).expand(B, N, L)

        pooled = torch.zeros(B, K, L, device=z_2d.device, dtype=z_2d.dtype)
        pooled.scatter_add_(1, node_idx, z_2d)

        counts = torch.zeros(B, K, 1, device=z_2d.device, dtype=z_2d.dtype)
        counts.scatter_add_(
            1,
            cluster_idx.view(1, N, 1).expand(B, N, 1),
            torch.ones(B, N, 1, device=z_2d.device, dtype=z_2d.dtype),
        )
        pooled = pooled / counts.clamp(min=1.0)

        coarse_edge_index = self.coarse_edge_index.to(z_2d.device)
        src = coarse_edge_index[0]
        dst = coarse_edge_index[1]
        src_feat = pooled[:, src, :]

        agg = torch.zeros(B, K, L, device=z_2d.device, dtype=z_2d.dtype)
        agg.scatter_add_(1, dst.view(1, -1, 1).expand(B, -1, L), src_feat)

        deg = torch.zeros(B, K, 1, device=z_2d.device, dtype=z_2d.dtype)
        deg.scatter_add_(
            1,
            dst.view(1, -1, 1).expand(B, -1, 1),
            torch.ones(B, src.shape[0], 1, device=z_2d.device, dtype=z_2d.dtype),
        )
        agg = agg / deg.clamp(min=1.0)

        pooled_refined = pooled + self.coarse_mlp(torch.cat([pooled, agg], dim=-1))
        coarse_context = pooled_refined[:, cluster_idx, :]
        fuse = torch.cat([z_2d, coarse_context], dim=-1)
        gate = torch.sigmoid(self.gate(fuse))
        return z_2d + gate * self.node_proj(fuse)


class LatentTransition(nn.Module):
    """
    Transition model for temporal latent z_t.

    z_{t+1} = z_t + f(z_t, u_t, c_e, graph)

    Where:
    - z_t: Current temporal latent
    - u_t: Control input (rainfall, inlet flow)
    - c_e: Event latent (conditioning)
    - graph: Spatial structure via GNN

    This is the core of VGSSM's temporal dynamics.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        control_dim: int = 2,
        step_scale: float = 0.1,
        use_tanh_update: bool = True,
        use_moe: bool = False,
        num_experts: int = 4,
        regime_hidden_dim: int = 64,
        moe_mode: str = "dense",
        moe_top_k: int = 1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.control_dim = control_dim
        self.step_scale = step_scale
        self.use_tanh_update = use_tanh_update
        self.use_moe = use_moe
        self.num_experts = max(1, int(num_experts))
        self.moe_mode = moe_mode
        self.moe_top_k = max(1, int(moe_top_k))

        # Control input projection
        self.control_proj = nn.Linear(control_dim, latent_dim)

        # Event latent projection (broadcast to nodes)
        self.ce_proj = nn.Linear(latent_dim, latent_dim)

        def make_local_mlp() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(latent_dim * 3, hidden_dim),  # z_t, u_t, c_e
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, latent_dim),
            )

        if self.use_moe:
            if self.moe_mode not in {"dense", "topk"}:
                raise ValueError(f"Unsupported moe_mode='{self.moe_mode}'. Use 'dense' or 'topk'.")
            self.local_experts = nn.ModuleList([make_local_mlp() for _ in range(self.num_experts)])
            regime_dim = latent_dim + 2 * control_dim
            self.regime_gate = nn.Sequential(
                nn.Linear(regime_dim, regime_hidden_dim),
                nn.GELU(),
                nn.Linear(regime_hidden_dim, self.num_experts),
            )
        else:
            self.local_mlp = make_local_mlp()

        # Spatial transition via simple message passing
        self.neighbor_agg = nn.Linear(latent_dim, latent_dim)
        self.combine = nn.Linear(latent_dim * 2, latent_dim)

    def forward(
        self,
        z_t: torch.Tensor,
        u_t: torch.Tensor,
        c_e: torch.Tensor,
        edge_index: torch.Tensor,
        u_proj: Optional[torch.Tensor] = None,
        u_mean: Optional[torch.Tensor] = None,
        u_std: Optional[torch.Tensor] = None,
        inv_dst_deg: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            z_t: [batch, num_nodes, latent_dim]
            u_t: [batch, num_nodes, control_dim] (rainfall/inlet)
            c_e: [batch, latent_dim]
            edge_index: [2, num_edges]
        Returns:
            z_{t+1}: [batch, num_nodes, latent_dim]
        """
        B, N, L = z_t.shape

        # Project inputs (optionally precomputed by caller for full rollout).
        if u_proj is None:
            u_proj = self.control_proj(u_t)  # [B, N, L]
        c_e_proj = self.ce_proj(c_e).unsqueeze(1).expand(-1, N, -1)  # [B, N, L]

        # Local transition
        local_input = torch.cat([z_t, u_proj, c_e_proj], dim=-1)  # [B, N, 3L]
        if self.use_moe:
            if u_mean is None:
                u_mean = u_t.mean(dim=1)
            if u_std is None:
                u_std = u_t.std(dim=1, unbiased=False)
            regime_feat = torch.cat([
                c_e,
                u_mean,
                u_std,
            ], dim=-1)
            gate_logits = self.regime_gate(regime_feat)  # [B, E]
            # Sparse MoE path: route each batch element to top-k experts only.
            if self.moe_mode == "topk" and self.moe_top_k < self.num_experts:
                k = min(self.num_experts, self.moe_top_k)
                topk_vals, topk_idx = torch.topk(gate_logits, k=k, dim=-1)  # [B, k]
                topk_gate = F.softmax(topk_vals, dim=-1)
                delta_local = torch.zeros_like(z_t)

                for expert_id, expert in enumerate(self.local_experts):
                    mask = topk_idx == expert_id  # [B, k]
                    if not mask.any():
                        continue
                    sample_idx, slot_idx = mask.nonzero(as_tuple=True)
                    expert_input = local_input[sample_idx]  # [M, N, 3L]
                    expert_out = expert(expert_input)  # [M, N, L]
                    weight = topk_gate[sample_idx, slot_idx].to(expert_out.dtype).view(-1, 1, 1)
                    # index_add_ requires exact dtype match under AMP.
                    delta_local.index_add_(
                        0,
                        sample_idx,
                        (expert_out * weight).to(delta_local.dtype),
                    )
            else:
                gate = F.softmax(gate_logits, dim=-1)  # [B, E]
                # Memory-efficient dense MoE combine: avoid materializing [B, E, N, L] tensor.
                delta_local = torch.zeros_like(z_t)
                for i, expert in enumerate(self.local_experts):
                    weight_i = gate[:, i].view(B, 1, 1)
                    delta_local = delta_local + weight_i * expert(local_input)
        else:
            delta_local = self.local_mlp(local_input)  # [B, N, L]

        # Spatial transition (simple mean aggregation from neighbors), vectorized over batch.
        src_idx = edge_index[0]
        dst_idx = edge_index[1]

        src_features = z_t[:, src_idx, :]  # [B, E, L]

        agg = torch.zeros(B, N, L, device=z_t.device, dtype=z_t.dtype)
        dst_expand = dst_idx.view(1, -1, 1).expand(B, -1, L)
        agg.scatter_add_(1, dst_expand, src_features)
        if inv_dst_deg is not None:
            agg = agg * inv_dst_deg.view(1, N, 1)
        else:
            count = torch.zeros(B, N, 1, device=z_t.device, dtype=z_t.dtype)
            ones = torch.ones(B, src_idx.numel(), 1, device=z_t.device, dtype=z_t.dtype)
            count.scatter_add_(1, dst_idx.view(1, -1, 1).expand(B, -1, 1), ones)
            agg = agg / count.clamp(min=1.0)

        delta_spatial = self.neighbor_agg(agg)  # [B, N, L]

        # Combine local and spatial
        delta = self.combine(torch.cat([delta_local, delta_spatial], dim=-1))  # [B, N, L]
        if self.use_tanh_update:
            delta = torch.tanh(delta)

        # Residual update
        z_next = z_t + self.step_scale * delta

        return z_next


class LatentDecoder(nn.Module):
    """
    Decode latent z_t to water levels AND edge flows.

    This is the KEY difference from VGSSM:
    - VGSSM only decoded to water levels
    - We also decode to edge flows for supervised training
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        num_1d_nodes: int,
        num_2d_nodes: int,
        num_1d_edges: int,
        num_2d_edges: int,
        edge_dim_1d: int = 4,
        edge_dim_2d: int = 4,
        flow_first: bool = False,
        use_stable_rollout: bool = True,
        use_nodewise_1d_dynamics: bool = False,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.edge_dim_1d = edge_dim_1d
        self.edge_dim_2d = edge_dim_2d
        self.num_1d_nodes = num_1d_nodes
        self.num_2d_nodes = num_2d_nodes
        self.flow_first = flow_first
        self.use_stable_rollout = use_stable_rollout
        self.use_nodewise_1d_dynamics = use_nodewise_1d_dynamics

        # Water level decoders
        self.wl_decoder_1d = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.wl_decoder_2d = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # Flow decoders (from source and dest node latents)
        self.flow_decoder_1d = nn.Sequential(
            nn.Linear(latent_dim * 2 + edge_dim_1d, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.flow_decoder_2d = nn.Sequential(
            nn.Linear(latent_dim * 2 + edge_dim_2d, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        if self.flow_first:
            self.wl_residual_1d = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            self.wl_residual_2d = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            control_hidden = max(16, hidden_dim // 2)
            self.force_1d = nn.Sequential(
                nn.Linear(2, control_hidden),
                nn.GELU(),
                nn.Linear(control_hidden, 1),
            )
            self.force_2d = nn.Sequential(
                nn.Linear(1, control_hidden),
                nn.GELU(),
                nn.Linear(control_hidden, 1),
            )
            self.mass_scale_1d = nn.Parameter(torch.tensor(0.01))
            self.mass_scale_2d = nn.Parameter(torch.tensor(0.01))
            # exp(-softplus(damping_log_*)) is in (0, 1], giving dissipative carry-over.
            self.damping_log_1d = nn.Parameter(torch.tensor(-8.0))
            self.damping_log_2d = nn.Parameter(torch.tensor(-8.0))
            # Bound decoded edge flows to avoid runaway divergence.
            self.flow_limit_1d = nn.Parameter(torch.tensor(1000.0))
            self.flow_limit_2d = nn.Parameter(torch.tensor(1000.0))

            # Stabilize rollout at initialization: start from persistence + tiny corrections.
            nn.init.zeros_(self.wl_residual_1d[-1].weight)
            nn.init.zeros_(self.wl_residual_1d[-1].bias)
            nn.init.zeros_(self.wl_residual_2d[-1].weight)
            nn.init.zeros_(self.wl_residual_2d[-1].bias)
            nn.init.zeros_(self.force_1d[-1].weight)
            nn.init.zeros_(self.force_1d[-1].bias)
            nn.init.zeros_(self.force_2d[-1].weight)
            nn.init.zeros_(self.force_2d[-1].bias)

            # Optional node-wise adaptation for heterogeneous 1D recession dynamics.
            # Zero init keeps behavior identical to scalar dynamics at start.
            if self.use_nodewise_1d_dynamics:
                node_hidden = max(16, hidden_dim // 2)
                self.mass_delta_1d = nn.Sequential(
                    nn.Linear(latent_dim, node_hidden),
                    nn.GELU(),
                    nn.Linear(node_hidden, 1),
                )
                self.decay_delta_1d = nn.Sequential(
                    nn.Linear(latent_dim, node_hidden),
                    nn.GELU(),
                    nn.Linear(node_hidden, 1),
                )
                self.mass_delta_scale_1d = nn.Parameter(torch.tensor(0.1))
                self.decay_delta_scale_1d = nn.Parameter(torch.tensor(0.1))
                nn.init.zeros_(self.mass_delta_1d[-1].weight)
                nn.init.zeros_(self.mass_delta_1d[-1].bias)
                nn.init.zeros_(self.decay_delta_1d[-1].weight)
                nn.init.zeros_(self.decay_delta_1d[-1].bias)

    def precompute_flow_edge_terms(
        self,
        edge_attr_1d: torch.Tensor,
        edge_attr_2d: torch.Tensor,
        dtype: Optional[torch.dtype] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project static edge features through the first flow layer once per forward pass."""
        first_1d: nn.Linear = self.flow_decoder_1d[0]
        first_2d: nn.Linear = self.flow_decoder_2d[0]
        split_idx = self.latent_dim * 2

        edge_attr_1d = edge_attr_1d.to(dtype=dtype or edge_attr_1d.dtype)
        edge_attr_2d = edge_attr_2d.to(dtype=dtype or edge_attr_2d.dtype)

        edge_term_1d = F.linear(edge_attr_1d, first_1d.weight[:, split_idx:], first_1d.bias)
        edge_term_2d = F.linear(edge_attr_2d, first_2d.weight[:, split_idx:], first_2d.bias)
        return edge_term_1d, edge_term_2d

    def _decode_flow_with_edge_term(
        self,
        src_latent: torch.Tensor,
        dst_latent: torch.Tensor,
        edge_term: torch.Tensor,
        decoder: nn.Sequential,
    ) -> torch.Tensor:
        first: nn.Linear = decoder[0]
        split_idx = self.latent_dim * 2

        dyn_input = torch.cat([src_latent, dst_latent], dim=-1)
        hidden = F.linear(dyn_input, first.weight[:, :split_idx], None)
        hidden = hidden + edge_term.unsqueeze(0)
        hidden = decoder[1](hidden)
        hidden = decoder[2](hidden)
        hidden = decoder[3](hidden)
        return decoder[4](hidden)

    @staticmethod
    def _flow_divergence(flow: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Args:
            flow: [B, E]
            edge_index: [2, E]
        Returns:
            divergence: [B, N]
        """
        B, E = flow.shape
        src = edge_index[0]
        dst = edge_index[1]
        divergence = torch.zeros(B, num_nodes, device=flow.device, dtype=flow.dtype)
        divergence.scatter_add_(1, dst.view(1, E).expand(B, E), flow)
        divergence.scatter_add_(1, src.view(1, E).expand(B, E), -flow)
        return divergence

    def forward(
        self,
        z_1d: torch.Tensor,  # [batch, num_1d, latent_dim]
        z_2d: torch.Tensor,  # [batch, num_2d, latent_dim]
        edge_index_1d: torch.Tensor,
        edge_index_2d: torch.Tensor,
        edge_attr_1d: torch.Tensor,
        edge_attr_2d: torch.Tensor,
        flow_edge_term_1d: Optional[torch.Tensor] = None,
        flow_edge_term_2d: Optional[torch.Tensor] = None,
        prev_wl_1d: Optional[torch.Tensor] = None,
        prev_wl_2d: Optional[torch.Tensor] = None,
        control_1d: Optional[torch.Tensor] = None,
        control_2d: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Decode latent to water levels and flows.
        """
        B = z_1d.shape[0]

        # Edge flows (vectorized over batch)
        src_1d = z_1d[:, edge_index_1d[0], :]  # [B, E1, L]
        dst_1d = z_1d[:, edge_index_1d[1], :]  # [B, E1, L]
        src_2d = z_2d[:, edge_index_2d[0], :]  # [B, E2, L]
        dst_2d = z_2d[:, edge_index_2d[1], :]  # [B, E2, L]

        if flow_edge_term_1d is not None and flow_edge_term_2d is not None:
            flow_1d = self._decode_flow_with_edge_term(src_1d, dst_1d, flow_edge_term_1d, self.flow_decoder_1d)
            flow_2d = self._decode_flow_with_edge_term(src_2d, dst_2d, flow_edge_term_2d, self.flow_decoder_2d)
        else:
            edge_attr_1d = edge_attr_1d.unsqueeze(0).expand(B, -1, -1)
            edge_attr_2d = edge_attr_2d.unsqueeze(0).expand(B, -1, -1)

            flow_input_1d = torch.cat([src_1d, dst_1d, edge_attr_1d], dim=-1)
            flow_input_2d = torch.cat([src_2d, dst_2d, edge_attr_2d], dim=-1)

            flow_1d = self.flow_decoder_1d(flow_input_1d)  # [B, E1, 1]
            flow_2d = self.flow_decoder_2d(flow_input_2d)  # [B, E2, 1]

        if self.flow_first and self.use_stable_rollout:
            flow_limit_1d = F.softplus(self.flow_limit_1d).to(device=flow_1d.device, dtype=flow_1d.dtype) + 1e-4
            flow_limit_2d = F.softplus(self.flow_limit_2d).to(device=flow_2d.device, dtype=flow_2d.dtype) + 1e-4
            flow_1d = torch.where(torch.isfinite(flow_1d), flow_1d, torch.zeros_like(flow_1d))
            flow_2d = torch.where(torch.isfinite(flow_2d), flow_2d, torch.zeros_like(flow_2d))
            flow_1d = flow_1d.clamp(min=-6.0 * flow_limit_1d, max=6.0 * flow_limit_1d)
            flow_2d = flow_2d.clamp(min=-6.0 * flow_limit_2d, max=6.0 * flow_limit_2d)
            flow_1d = flow_limit_1d * torch.tanh(flow_1d / flow_limit_1d)
            flow_2d = flow_limit_2d * torch.tanh(flow_2d / flow_limit_2d)

        if self.flow_first and prev_wl_1d is not None and prev_wl_2d is not None:
            div_1d = self._flow_divergence(flow_1d[..., 0], edge_index_1d, self.num_1d_nodes).unsqueeze(-1)
            div_2d = self._flow_divergence(flow_2d[..., 0], edge_index_2d, self.num_2d_nodes).unsqueeze(-1)

            if control_1d is None:
                control_1d = torch.zeros(B, self.num_1d_nodes, 2, device=z_1d.device, dtype=z_1d.dtype)
            if control_2d is None:
                control_2d = torch.zeros(B, self.num_2d_nodes, 1, device=z_2d.device, dtype=z_2d.dtype)

            force_1d = self.force_1d(control_1d)
            force_2d = self.force_2d(control_2d)
            residual_1d = self.wl_residual_1d(z_1d)
            residual_2d = self.wl_residual_2d(z_2d)
            if self.use_stable_rollout:
                mass_gain_1d = torch.tanh(self.mass_scale_1d).to(device=z_1d.device, dtype=z_1d.dtype)
                mass_gain_2d = torch.tanh(self.mass_scale_2d).to(device=z_2d.device, dtype=z_2d.dtype)
                decay_1d = torch.exp(-F.softplus(self.damping_log_1d)).to(device=z_1d.device, dtype=z_1d.dtype)
                decay_2d = torch.exp(-F.softplus(self.damping_log_2d)).to(device=z_2d.device, dtype=z_2d.dtype)

                if self.use_nodewise_1d_dynamics:
                    mass_delta = torch.tanh(self.mass_delta_1d(z_1d))
                    mass_delta_scale = torch.tanh(self.mass_delta_scale_1d).to(device=z_1d.device, dtype=z_1d.dtype)
                    mass_gain_1d = torch.clamp(mass_gain_1d + mass_delta * mass_delta_scale, min=-0.95, max=0.95)

                    decay_delta = torch.tanh(self.decay_delta_1d(z_1d))
                    decay_delta_scale = torch.tanh(self.decay_delta_scale_1d).to(device=z_1d.device, dtype=z_1d.dtype)
                    decay_1d = torch.clamp(decay_1d + decay_delta * decay_delta_scale, min=0.0, max=1.0)

                wl_1d = (
                    decay_1d * prev_wl_1d
                    + mass_gain_1d * div_1d
                    + force_1d
                    + residual_1d
                )
                wl_2d = (
                    decay_2d * prev_wl_2d
                    + mass_gain_2d * div_2d
                    + force_2d
                    + residual_2d
                )
            else:
                mass_gain_1d = torch.tanh(self.mass_scale_1d).to(device=z_1d.device, dtype=z_1d.dtype)
                prev_gain_1d: torch.Tensor | float = 1.0
                if self.use_nodewise_1d_dynamics:
                    mass_delta = torch.tanh(self.mass_delta_1d(z_1d))
                    mass_delta_scale = torch.tanh(self.mass_delta_scale_1d).to(device=z_1d.device, dtype=z_1d.dtype)
                    mass_gain_1d = torch.clamp(mass_gain_1d + mass_delta * mass_delta_scale, min=-0.95, max=0.95)

                    # When stable rollout is disabled, use decay head as a bounded
                    # multiplicative adjustment around persistence (gain=1).
                    decay_delta = torch.tanh(self.decay_delta_1d(z_1d))
                    decay_delta_scale = torch.tanh(self.decay_delta_scale_1d).to(device=z_1d.device, dtype=z_1d.dtype)
                    prev_gain_1d = torch.clamp(1.0 + decay_delta * decay_delta_scale, min=0.5, max=1.5)

                wl_1d = prev_gain_1d * prev_wl_1d + mass_gain_1d * div_1d + force_1d + residual_1d
                wl_2d = prev_wl_2d + torch.tanh(self.mass_scale_2d) * div_2d + force_2d + residual_2d
        else:
            # Direct decode fallback.
            wl_1d = self.wl_decoder_1d(z_1d)  # [B, N1, 1]
            wl_2d = self.wl_decoder_2d(z_2d)  # [B, N2, 1]

        return {
            'wl_1d': wl_1d,
            'wl_2d': wl_2d,
            'flow_1d': flow_1d,
            'flow_2d': flow_2d,
        }


# ==============================================================================
# MODEL: DualFloodGNN
# ==============================================================================

class NodeEncoder(nn.Module):
    """Encode node features + dynamic state into latent representation."""

    def __init__(self, static_dim: int, dynamic_dim: int, hidden_dim: int):
        super().__init__()
        self.static_proj = nn.Linear(static_dim, hidden_dim)
        self.dynamic_proj = nn.Linear(dynamic_dim, hidden_dim)
        self.combine = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, static: torch.Tensor, dynamic: torch.Tensor) -> torch.Tensor:
        """
        Args:
            static: [num_nodes, static_dim]
            dynamic: [batch, num_nodes, dynamic_dim]
        Returns:
            [batch, num_nodes, hidden_dim]
        """
        static_emb = self.static_proj(static)  # [N, H]
        dynamic_emb = self.dynamic_proj(dynamic)  # [B, N, H]

        # Expand static to batch
        static_emb = static_emb.unsqueeze(0).expand(dynamic_emb.shape[0], -1, -1)

        combined = torch.cat([static_emb, dynamic_emb], dim=-1)
        return self.combine(combined)


class EdgeEncoder(nn.Module):
    """Encode edge features into latent representation."""

    def __init__(self, edge_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(edge_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.proj(edge_attr)


class MessagePassingBlock(nn.Module):
    """Heterogeneous message passing for 1D-2D coupled network."""

    def __init__(self, hidden_dim: int):
        super().__init__()

        # Message functions for each edge type
        self.conv = HeteroConv({
            ('1d', 'pipe', '1d'): SAGEConv(hidden_dim, hidden_dim),
            ('2d', 'surface', '2d'): SAGEConv(hidden_dim, hidden_dim),
            ('1d', 'couples_to', '2d'): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
            ('2d', 'couples_from', '1d'): SAGEConv((hidden_dim, hidden_dim), hidden_dim),
        }, aggr='sum')

        self.norm_1d = nn.LayerNorm(hidden_dim)
        self.norm_2d = nn.LayerNorm(hidden_dim)

        self.ff_1d = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.ff_2d = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x_dict: Dict[str, torch.Tensor], edge_index_dict: Dict) -> Dict[str, torch.Tensor]:
        # Message passing
        out = self.conv(x_dict, edge_index_dict)

        # Residual + LayerNorm + FFN for each node type
        out['1d'] = self.norm_1d(x_dict['1d'] + out['1d'])
        out['1d'] = out['1d'] + self.ff_1d(out['1d'])

        out['2d'] = self.norm_2d(x_dict['2d'] + out['2d'])
        out['2d'] = out['2d'] + self.ff_2d(out['2d'])

        return out


class FlowPredictor(nn.Module):
    """Predict edge flows from node embeddings.

    This is the KEY component - predicts flow on each edge using
    source and destination node embeddings.
    """

    def __init__(self, hidden_dim: int, edge_dim: int = 0):
        super().__init__()
        # Flow depends on source node, dest node, and edge features
        input_dim = hidden_dim * 2 + edge_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),  # Predict single flow value
        )

    def forward(self, h_src: torch.Tensor, h_dst: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            h_src: [num_edges, hidden_dim] - source node embeddings
            h_dst: [num_edges, hidden_dim] - destination node embeddings
            edge_attr: [num_edges, edge_dim] - edge features (optional)
        Returns:
            flow: [num_edges, 1] - predicted flow (signed)
        """
        if edge_attr is not None:
            x = torch.cat([h_src, h_dst, edge_attr], dim=-1)
        else:
            x = torch.cat([h_src, h_dst], dim=-1)
        return self.mlp(x)


class TemporalBlock(nn.Module):
    """Process temporal sequences with attention."""

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, hidden_dim]
        Returns:
            [batch, seq_len, hidden_dim]
        """
        # Self-attention
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)

        # FFN
        x = self.norm2(x + self.ff(x))
        return x


class DualFloodGNN(nn.Module):
    """
    DualFloodGNN v2: Latent-space flood prediction with supervised edge flows.

    Architecture:
    1. PrefixEncoder: GRU over prefix sequence -> h_prefix
    2. EventLatentInference: h_prefix -> c_e (event latent)
    3. TemporalLatentInference: h_prefix, c_e -> z_0 (initial temporal latent)
    4. LatentTransition: z_t, u_t, c_e -> z_{t+1} (rollout dynamics)
    5. LatentDecoder: z_t -> water levels AND edge flows

    Key insight: Combines VGSSM's powerful latent dynamics with supervised edge flows.
    """

    def __init__(
        self,
        num_1d_nodes: int,
        num_2d_nodes: int,
        num_1d_edges: int,
        num_2d_edges: int,
        static_dim_1d: int = 6,
        static_dim_2d: int = 9,
        dynamic_dim_1d: int = 2,  # water_level, inlet_flow
        dynamic_dim_2d: int = 3,  # water_level, rainfall, water_volume
        edge_dim_1d: int = 4,
        edge_dim_2d: int = 4,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        num_gnn_layers: int = 3,
        num_temporal_layers: int = 2,
        pred_len: int = 90,
        transition_scale: float = 0.1,
        coupling_scale: float = 0.1,
        use_flow_first_decoder: bool = False,
        use_multiscale_2d: bool = False,
        multiscale_num_clusters: int = 128,
        use_moe_transition: bool = False,
        moe_num_experts: int = 4,
        moe_mode: str = "dense",
        moe_top_k: int = 1,
        use_dual_timescale_latent: bool = False,
        slow_timescale_ratio: float = 0.25,
        use_direct_ar_hybrid: bool = False,
        direct_ar_init_blend: float = 0.5,
        precompute_transition_controls: bool = True,
        precompute_decoder_edge_terms: bool = True,
        use_stable_flow_rollout: bool = True,
        use_inlet_imputer: bool = False,
        use_nodewise_1d_dynamics: bool = False,
    ):
        super().__init__()

        self.num_1d_nodes = num_1d_nodes
        self.num_2d_nodes = num_2d_nodes
        self.num_1d_edges = num_1d_edges
        self.num_2d_edges = num_2d_edges
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.pred_len = pred_len
        self.coupling_scale = coupling_scale
        self.use_flow_first_decoder = use_flow_first_decoder
        self.use_multiscale_2d = use_multiscale_2d
        self.use_dual_timescale_latent = use_dual_timescale_latent
        self.use_direct_ar_hybrid = use_direct_ar_hybrid
        self.precompute_transition_controls = precompute_transition_controls
        self.precompute_decoder_edge_terms = precompute_decoder_edge_terms
        self.use_stable_flow_rollout = use_stable_flow_rollout
        self.use_inlet_imputer = use_inlet_imputer
        self.use_nodewise_1d_dynamics = use_nodewise_1d_dynamics
        self.use_moe_transition = use_moe_transition
        self.moe_mode = moe_mode
        self.moe_top_k = max(1, int(moe_top_k))

        # Static feature encoders (embed static features)
        self.static_encoder_1d = nn.Sequential(
            nn.Linear(static_dim_1d, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.static_encoder_2d = nn.Sequential(
            nn.Linear(static_dim_2d, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Prefix encoders (GRU over dynamic features)
        # Input: static_emb + dynamic features
        self.prefix_encoder_1d = PrefixEncoder(
            input_dim=hidden_dim + dynamic_dim_1d,
            hidden_dim=hidden_dim,
            num_layers=2,
        )
        self.prefix_encoder_2d = PrefixEncoder(
            input_dim=hidden_dim + dynamic_dim_2d,
            hidden_dim=hidden_dim,
            num_layers=2,
        )

        # Event latent inference (from combined 1d+2d prefix)
        self.event_latent_inference = EventLatentInference(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
        )

        # Temporal latent inference (per node type)
        self.temporal_latent_1d = TemporalLatentInference(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_nodes=num_1d_nodes,
        )
        self.temporal_latent_2d = TemporalLatentInference(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            num_nodes=num_2d_nodes,
        )

        # Latent transition models
        self.transition_1d = LatentTransition(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            control_dim=2,  # inlet_flow + inlet_missing_mask
            step_scale=transition_scale,
            use_tanh_update=True,
            use_moe=use_moe_transition,
            num_experts=moe_num_experts,
            moe_mode=moe_mode,
            moe_top_k=moe_top_k,
        )
        self.transition_2d = LatentTransition(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            control_dim=1,  # rainfall
            step_scale=transition_scale,
            use_tanh_update=True,
            use_moe=use_moe_transition,
            num_experts=moe_num_experts,
            moe_mode=moe_mode,
            moe_top_k=moe_top_k,
        )

        if self.use_dual_timescale_latent:
            slow_scale = max(1e-4, transition_scale * slow_timescale_ratio)
            self.transition_1d_slow = LatentTransition(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                control_dim=2,
                step_scale=slow_scale,
                use_tanh_update=True,
                use_moe=use_moe_transition,
                num_experts=moe_num_experts,
                moe_mode=moe_mode,
                moe_top_k=moe_top_k,
            )
            self.transition_2d_slow = LatentTransition(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                control_dim=1,
                step_scale=slow_scale,
                use_tanh_update=True,
                use_moe=use_moe_transition,
                num_experts=moe_num_experts,
                moe_mode=moe_mode,
                moe_top_k=moe_top_k,
            )
            self.slow_init_1d = nn.Linear(latent_dim, latent_dim)
            self.slow_init_2d = nn.Linear(latent_dim, latent_dim)
            self.latent_fuse_1d = nn.Sequential(
                nn.Linear(latent_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, latent_dim),
            )
            self.latent_fuse_2d = nn.Sequential(
                nn.Linear(latent_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, latent_dim),
            )

            # Start with negligible slow-path influence; learn it progressively.
            nn.init.zeros_(self.slow_init_1d.weight)
            nn.init.zeros_(self.slow_init_1d.bias)
            nn.init.zeros_(self.slow_init_2d.weight)
            nn.init.zeros_(self.slow_init_2d.bias)
            nn.init.zeros_(self.latent_fuse_1d[-1].weight)
            nn.init.zeros_(self.latent_fuse_1d[-1].bias)
            nn.init.zeros_(self.latent_fuse_2d[-1].weight)
            nn.init.zeros_(self.latent_fuse_2d[-1].bias)

        if self.use_multiscale_2d:
            self.multiscale_encoder_2d = MultiScale2DEncoder(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                num_clusters=multiscale_num_clusters,
            )

        # Edge attribute encoders
        self.edge_encoder_1d = EdgeEncoder(edge_dim_1d, hidden_dim)
        self.edge_encoder_2d = EdgeEncoder(edge_dim_2d, hidden_dim)

        # Store edge dims for decoder
        self.edge_dim_1d = edge_dim_1d
        self.edge_dim_2d = edge_dim_2d

        # Latent decoder (to water levels and flows)
        self.decoder = LatentDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            num_1d_nodes=num_1d_nodes,
            num_2d_nodes=num_2d_nodes,
            num_1d_edges=num_1d_edges,
            num_2d_edges=num_2d_edges,
            edge_dim_1d=edge_dim_1d,
            edge_dim_2d=edge_dim_2d,
            flow_first=use_flow_first_decoder,
            use_stable_rollout=use_stable_flow_rollout,
            use_nodewise_1d_dynamics=use_nodewise_1d_dynamics,
        )

        # Cross-network coupling (1D <-> 2D latent exchange)
        self.coupling_1d_to_2d = nn.Linear(latent_dim, latent_dim)
        self.coupling_2d_to_1d = nn.Linear(latent_dim, latent_dim)

        # Auxiliary imputer for unknown future inlet controls.
        # Learns delta inlet from latent state, event embedding, and previous inlet.
        if self.use_inlet_imputer:
            inlet_hidden = max(32, hidden_dim // 2)
            self.inlet_imputer = nn.Sequential(
                nn.Linear(latent_dim * 2 + 4, inlet_hidden),
                nn.GELU(),
                nn.Linear(inlet_hidden, 1),
            )
            # Bound per-step inlet correction and absolute level in normalized space.
            self.inlet_delta_scale = nn.Parameter(torch.tensor(0.05))
            self.inlet_level_limit = nn.Parameter(torch.tensor(6.0))
            self.inlet_state_decay_logit = nn.Parameter(torch.tensor(2.0))
            self.inlet_rain_gain = nn.Parameter(torch.tensor(0.05))
            self.inlet_prev_gain = nn.Parameter(torch.tensor(0.05))
            self.inlet_state_limit = nn.Parameter(torch.tensor(6.0))
            nn.init.zeros_(self.inlet_imputer[-1].weight)
            nn.init.zeros_(self.inlet_imputer[-1].bias)

        if self.use_direct_ar_hybrid:
            self.direct_time_proj = nn.Sequential(
                nn.Linear(1, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, latent_dim),
            )
            self.direct_ce_proj = nn.Linear(latent_dim, latent_dim)
            self.direct_head_1d = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            self.direct_head_2d = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
            blend = float(np.clip(direct_ar_init_blend, 1e-3, 1.0 - 1e-3))
            blend_logit = math.log(blend / (1.0 - blend))
            self.direct_ar_blend_logit = nn.Parameter(torch.tensor(blend_logit, dtype=torch.float32))

            # Keep initial direct head conservative when loaded from non-hybrid checkpoints.
            nn.init.zeros_(self.direct_head_1d[-1].weight)
            nn.init.zeros_(self.direct_head_1d[-1].bias)
            nn.init.zeros_(self.direct_head_2d[-1].weight)
            nn.init.zeros_(self.direct_head_2d[-1].bias)

        self._cached_graph_device: Optional[str] = None
        self._cached_graph_tensors: Dict[str, torch.Tensor] = {}

    def _get_graph_tensors(self, graph: HeteroData, device: torch.device) -> Dict[str, torch.Tensor]:
        device_key = str(device)
        if device_key != self._cached_graph_device:
            static_1d = graph['1d'].x.to(device, non_blocking=True)
            static_2d = graph['2d'].x.to(device, non_blocking=True)
            edge_index_1d = graph['1d', 'pipe', '1d'].edge_index.to(device, non_blocking=True)
            edge_index_2d = graph['2d', 'surface', '2d'].edge_index.to(device, non_blocking=True)
            edge_attr_1d = graph['1d', 'pipe', '1d'].edge_attr.to(device, non_blocking=True)
            edge_attr_2d = graph['2d', 'surface', '2d'].edge_attr.to(device, non_blocking=True)
            coupling_1d_2d = graph['1d', 'couples_to', '2d'].edge_index.to(device, non_blocking=True)

            src_1d = coupling_1d_2d[0]
            dst_2d = coupling_1d_2d[1]
            deg_1d = torch.bincount(src_1d, minlength=static_1d.shape[0]).to(device=device, dtype=torch.float32).clamp(min=1.0)
            deg_2d = torch.bincount(dst_2d, minlength=static_2d.shape[0]).to(device=device, dtype=torch.float32).clamp(min=1.0)
            dst_deg_1d = torch.bincount(edge_index_1d[1], minlength=static_1d.shape[0]).to(
                device=device, dtype=torch.float32
            ).clamp(min=1.0)
            dst_deg_2d = torch.bincount(edge_index_2d[1], minlength=static_2d.shape[0]).to(
                device=device, dtype=torch.float32
            ).clamp(min=1.0)
            inv_dst_deg_1d = dst_deg_1d.reciprocal()
            inv_dst_deg_2d = dst_deg_2d.reciprocal()

            self._cached_graph_tensors = {
                'static_1d': static_1d,
                'static_2d': static_2d,
                'edge_index_1d': edge_index_1d,
                'edge_index_2d': edge_index_2d,
                'edge_attr_1d': edge_attr_1d,
                'edge_attr_2d': edge_attr_2d,
                'coupling_1d_2d': coupling_1d_2d,
                'src_1d': src_1d,
                'dst_2d': dst_2d,
                'deg_1d': deg_1d,
                'deg_2d': deg_2d,
                'inv_dst_deg_1d': inv_dst_deg_1d,
                'inv_dst_deg_2d': inv_dst_deg_2d,
            }
            self._cached_graph_device = device_key
        return self._cached_graph_tensors

    def _fuse_latent(self, z_fast: torch.Tensor, z_slow: torch.Tensor, kind: str) -> torch.Tensor:
        if kind == '1d':
            return z_fast + self.latent_fuse_1d(torch.cat([z_fast, z_slow], dim=-1))
        return z_fast + self.latent_fuse_2d(torch.cat([z_fast, z_slow], dim=-1))

    def _apply_cross_coupling(
        self,
        z_1d: torch.Tensor,
        z_2d: torch.Tensor,
        src_1d: torch.Tensor,
        dst_2d: torch.Tensor,
        deg_1d: torch.Tensor,
        deg_2d: torch.Tensor,
        idx_1d: torch.Tensor,
        idx_2d: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        src_lat_1d = z_1d[:, src_1d, :]
        src_lat_2d = z_2d[:, dst_2d, :]
        coupling_1d_to_2d = torch.tanh(self.coupling_1d_to_2d(src_lat_1d)) / deg_2d[dst_2d].view(1, -1, 1)
        coupling_2d_to_1d = torch.tanh(self.coupling_2d_to_1d(src_lat_2d)) / deg_1d[src_1d].view(1, -1, 1)
        coupling_1d_to_2d = coupling_1d_to_2d.to(z_2d.dtype)
        coupling_2d_to_1d = coupling_2d_to_1d.to(z_1d.dtype)
        z_2d.scatter_add_(1, idx_2d, self.coupling_scale * coupling_1d_to_2d)
        z_1d.scatter_add_(1, idx_1d, self.coupling_scale * coupling_2d_to_1d)
        return z_1d, z_2d

    @staticmethod
    def _aggregate_coupled_signal(
        signal_2d: torch.Tensor,
        src_1d: torch.Tensor,
        dst_2d: torch.Tensor,
        deg_1d: torch.Tensor,
    ) -> torch.Tensor:
        """
        Aggregate 2D node signals onto 1D nodes through coupling edges.

        Args:
            signal_2d: [B, N2, C]
            src_1d: [E_c] 1D node index per coupling edge
            dst_2d: [E_c] 2D node index per coupling edge
            deg_1d: [N1] coupling count per 1D node
        Returns:
            [B, N1, C] mean coupled signal for each 1D node
        """
        B, _, C = signal_2d.shape
        n1 = int(deg_1d.shape[0])
        out = torch.zeros(B, n1, C, device=signal_2d.device, dtype=signal_2d.dtype)
        gathered = signal_2d[:, dst_2d, :]
        idx = src_1d.view(1, -1, 1).expand(B, -1, C)
        out.scatter_add_(1, idx, gathered)
        denom = deg_1d.to(device=signal_2d.device, dtype=signal_2d.dtype).view(1, n1, 1).clamp_min(1.0)
        return out / denom

    def _build_direct_predictions(
        self,
        z0_1d: torch.Tensor,
        z0_2d: torch.Tensor,
        c_e: torch.Tensor,
        rollout_len: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N1, _ = z0_1d.shape
        _, N2, _ = z0_2d.shape
        t = torch.arange(rollout_len, device=z0_1d.device, dtype=z0_1d.dtype)
        denom = max(rollout_len - 1, 1)
        t = (t / denom).view(1, rollout_len, 1, 1)
        t_emb = self.direct_time_proj(t)
        ce_emb = self.direct_ce_proj(c_e).view(B, 1, 1, self.latent_dim)
        direct_in_1d = z0_1d.unsqueeze(1).expand(B, rollout_len, N1, self.latent_dim) + t_emb + ce_emb
        direct_in_2d = z0_2d.unsqueeze(1).expand(B, rollout_len, N2, self.latent_dim) + t_emb + ce_emb
        return self.direct_head_1d(direct_in_1d), self.direct_head_2d(direct_in_2d)

    def forward(
        self,
        graph: HeteroData,
        input_1d: torch.Tensor,  # [batch, seq_len, num_1d, dynamic_dim]
        input_2d: torch.Tensor,  # [batch, seq_len, num_2d, dynamic_dim]
        future_rainfall: torch.Tensor,  # [batch, pred_len, num_2d, 1]
        future_inlet: torch.Tensor,  # [batch, pred_len, num_1d, 1]
        future_inlet_mask: Optional[torch.Tensor] = None,  # [batch, pred_len, num_1d, 1]
        rollout_len: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with latent space dynamics.
        """
        B, T_prefix, N1, _ = input_1d.shape
        _, _, N2, _ = input_2d.shape
        device = input_1d.device
        T = rollout_len if rollout_len is not None else self.pred_len
        if future_inlet_mask is None:
            future_inlet_mask = torch.ones_like(future_inlet)

        graph_tensors = self._get_graph_tensors(graph, device)

        # Get static features
        static_1d = graph_tensors['static_1d']  # [N1, static_dim]
        static_2d = graph_tensors['static_2d']  # [N2, static_dim]

        # Get edge info
        edge_index_1d = graph_tensors['edge_index_1d']
        edge_index_2d = graph_tensors['edge_index_2d']
        edge_attr_1d = graph_tensors['edge_attr_1d']
        edge_attr_2d = graph_tensors['edge_attr_2d']

        # Get 1D-2D coupling edges
        coupling_1d_2d = graph_tensors['coupling_1d_2d']

        if self.use_multiscale_2d:
            if (
                self.multiscale_encoder_2d.cluster_idx.numel() != N2
                or self.multiscale_encoder_2d.cluster_idx.device != static_2d.device
            ):
                self.multiscale_encoder_2d.set_graph(static_2d[:, :2].detach(), edge_index_2d)

        # === STEP 1: Encode static features ===
        # Static coordinates are large (order 1e6); force FP32 here to avoid fp16 overflow under AMP.
        if torch.is_autocast_enabled() and static_1d.is_cuda:
            with torch.autocast(device_type='cuda', enabled=False):
                static_emb_1d = self.static_encoder_1d(static_1d.float())  # [N1, H]
                static_emb_2d = self.static_encoder_2d(static_2d.float())  # [N2, H]
        else:
            static_emb_1d = self.static_encoder_1d(static_1d)  # [N1, H]
            static_emb_2d = self.static_encoder_2d(static_2d)  # [N2, H]

        # Expand to batch
        static_emb_1d = static_emb_1d.unsqueeze(0).unsqueeze(0).expand(B, T_prefix, -1, -1)  # [B, T, N1, H]
        static_emb_2d = static_emb_2d.unsqueeze(0).unsqueeze(0).expand(B, T_prefix, -1, -1)  # [B, T, N2, H]

        # === STEP 2: Encode prefix with GRU ===
        # Concatenate static + dynamic
        prefix_input_1d = torch.cat([static_emb_1d, input_1d], dim=-1)  # [B, T, N1, H+D1]
        prefix_input_2d = torch.cat([static_emb_2d, input_2d], dim=-1)  # [B, T, N2, H+D2]

        h_prefix_1d = self.prefix_encoder_1d(prefix_input_1d)  # [B, N1, H]
        h_prefix_2d = self.prefix_encoder_2d(prefix_input_2d)  # [B, N2, H]

        # === STEP 3: Infer event latent c_e ===
        # Combine 1D and 2D prefix info
        h_prefix_combined = torch.cat([
            h_prefix_1d.mean(dim=1, keepdim=True),  # [B, 1, H]
            h_prefix_2d.mean(dim=1, keepdim=True),  # [B, 1, H]
        ], dim=1)  # [B, 2, H]
        c_e, mu_ce, logvar_ce = self.event_latent_inference(h_prefix_combined)

        # === STEP 4: Infer initial temporal latent z_0 ===
        z_0_1d, mu_z0_1d, logvar_z0_1d = self.temporal_latent_1d(h_prefix_1d, c_e)  # [B, N1, L]
        z_0_2d, mu_z0_2d, logvar_z0_2d = self.temporal_latent_2d(h_prefix_2d, c_e)  # [B, N2, L]

        # === STEP 5: Rollout with transition model ===
        if self.use_dual_timescale_latent:
            z_fast_1d = z_0_1d
            z_fast_2d = z_0_2d
            z_slow_1d = self.slow_init_1d(z_0_1d)
            z_slow_2d = self.slow_init_2d(z_0_2d)
            z_1d = self._fuse_latent(z_fast_1d, z_slow_1d, kind='1d')
            z_2d = self._fuse_latent(z_fast_2d, z_slow_2d, kind='2d')
        else:
            z_1d = z_0_1d
            z_2d = z_0_2d

        all_wl_1d = []
        all_wl_2d = []
        all_flow_1d = []
        all_flow_2d = []

        wl_state_1d = input_1d[:, -1, :, 0:1]
        wl_state_2d = input_2d[:, -1, :, 0:1]

        src_1d = graph_tensors['src_1d']
        dst_2d = graph_tensors['dst_2d']
        deg_1d = graph_tensors['deg_1d']
        deg_2d = graph_tensors['deg_2d']
        inv_dst_deg_1d = graph_tensors['inv_dst_deg_1d'].to(device=device, dtype=z_1d.dtype)
        inv_dst_deg_2d = graph_tensors['inv_dst_deg_2d'].to(device=device, dtype=z_2d.dtype)
        idx_1d = src_1d.view(1, -1, 1).expand(B, -1, self.latent_dim)
        idx_2d = dst_2d.view(1, -1, 1).expand(B, -1, self.latent_dim)

        u_1d_all = None
        u_2d_all = None
        u1d_fast_proj_all = None
        u2d_fast_proj_all = None
        u1d_fast_mean_all = None
        u1d_fast_std_all = None
        u2d_fast_mean_all = None
        u2d_fast_std_all = None
        u1d_slow_proj_all = None
        u2d_slow_proj_all = None
        u1d_slow_mean_all = None
        u1d_slow_std_all = None
        u2d_slow_mean_all = None
        u2d_slow_std_all = None

        precompute_transition = self.precompute_transition_controls and not self.use_inlet_imputer
        if precompute_transition:
            u_1d_all = torch.cat([future_inlet[:, :T], future_inlet_mask[:, :T]], dim=-1).contiguous()
            u_2d_all = future_rainfall[:, :T].contiguous()

            u1d_fast_proj_all = self.transition_1d.control_proj(u_1d_all)
            u2d_fast_proj_all = self.transition_2d.control_proj(u_2d_all)

            if self.transition_1d.use_moe:
                u1d_fast_mean_all = u_1d_all.mean(dim=2)
                u1d_fast_std_all = u_1d_all.std(dim=2, unbiased=False)
            if self.transition_2d.use_moe:
                u2d_fast_mean_all = u_2d_all.mean(dim=2)
                u2d_fast_std_all = u_2d_all.std(dim=2, unbiased=False)

            if self.use_dual_timescale_latent:
                u1d_slow_proj_all = self.transition_1d_slow.control_proj(u_1d_all)
                u2d_slow_proj_all = self.transition_2d_slow.control_proj(u_2d_all)
                if self.transition_1d_slow.use_moe:
                    u1d_slow_mean_all = u_1d_all.mean(dim=2)
                    u1d_slow_std_all = u_1d_all.std(dim=2, unbiased=False)
                if self.transition_2d_slow.use_moe:
                    u2d_slow_mean_all = u_2d_all.mean(dim=2)
                    u2d_slow_std_all = u_2d_all.std(dim=2, unbiased=False)

        flow_edge_term_1d = None
        flow_edge_term_2d = None
        if self.precompute_decoder_edge_terms:
            flow_edge_term_1d, flow_edge_term_2d = self.decoder.precompute_flow_edge_terms(
                edge_attr_1d,
                edge_attr_2d,
                dtype=z_1d.dtype,
            )

        prev_inlet = input_1d[:, -1, :, 1:2]
        inlet_state = prev_inlet.clone() if self.use_inlet_imputer else None
        ce_node_1d = c_e.unsqueeze(1).expand(-1, N1, -1) if self.use_inlet_imputer else None
        all_inlet_pred_1d = [] if self.use_inlet_imputer else None

        for t in range(T):
            u_2d = future_rainfall[:, t, :, :]  # [B, N2, 1]
            if self.use_inlet_imputer:
                inlet_obs_t = future_inlet[:, t, :, :]  # [B, N1, 1]
                inlet_mask_t = future_inlet_mask[:, t, :, :]  # [B, N1, 1]

                # Coupling-aware rainfall forcing with bounded residual state.
                rain_to_1d = self._aggregate_coupled_signal(
                    u_2d,
                    src_1d,
                    dst_2d,
                    deg_1d,
                )
                decay = torch.sigmoid(self.inlet_state_decay_logit).to(device=z_1d.device, dtype=z_1d.dtype)
                rain_gain = F.softplus(self.inlet_rain_gain).to(device=z_1d.device, dtype=z_1d.dtype) + 1e-4
                prev_gain = torch.tanh(self.inlet_prev_gain).to(device=z_1d.device, dtype=z_1d.dtype)
                inlet_state = decay * inlet_state + rain_gain * rain_to_1d + prev_gain * prev_inlet
                state_limit = F.softplus(self.inlet_state_limit).to(device=z_1d.device, dtype=z_1d.dtype) + 1e-4
                inlet_state = state_limit * torch.tanh(inlet_state / state_limit)
                inlet_state = torch.where(torch.isfinite(inlet_state), inlet_state, prev_inlet)

                inlet_delta_raw_t = self.inlet_imputer(
                    torch.cat(
                        [z_1d, ce_node_1d, prev_inlet, inlet_mask_t, rain_to_1d, inlet_state],
                        dim=-1,
                    )
                )
                delta_scale = F.softplus(self.inlet_delta_scale).to(device=z_1d.device, dtype=z_1d.dtype) + 1e-4
                inlet_delta_t = delta_scale * torch.tanh(inlet_delta_raw_t)
                inlet_pred_t = inlet_state + inlet_delta_t
                level_limit = F.softplus(self.inlet_level_limit).to(device=z_1d.device, dtype=z_1d.dtype) + 1e-4
                inlet_pred_t = level_limit * torch.tanh(inlet_pred_t / level_limit)
                inlet_pred_t = torch.where(torch.isfinite(inlet_pred_t), inlet_pred_t, prev_inlet)
                inlet_used_t = inlet_mask_t * inlet_obs_t + (1.0 - inlet_mask_t) * inlet_pred_t
                all_inlet_pred_1d.append(inlet_pred_t)
                prev_inlet = inlet_used_t
                u_1d = torch.cat([inlet_used_t, inlet_mask_t], dim=-1)  # [B, N1, 2]
            elif u_1d_all is not None and u_2d_all is not None:
                u_1d = u_1d_all[:, t, :, :]  # [B, N1, 2]
                u_2d = u_2d_all[:, t, :, :]  # [B, N2, 1]
            else:
                inlet_t = future_inlet[:, t, :, :]  # [B, N1, 1]
                inlet_mask_t = future_inlet_mask[:, t, :, :]  # [B, N1, 1]
                u_1d = torch.cat([inlet_t, inlet_mask_t], dim=-1)  # [B, N1, 2]
                u_2d = future_rainfall[:, t, :, :]  # [B, N2, 1]

            z_decode_2d = self.multiscale_encoder_2d(z_2d) if self.use_multiscale_2d else z_2d

            # Decode current latent state.
            decoded = self.decoder(
                z_1d,
                z_decode_2d,
                edge_index_1d, edge_index_2d,
                edge_attr_1d, edge_attr_2d,
                flow_edge_term_1d=flow_edge_term_1d,
                flow_edge_term_2d=flow_edge_term_2d,
                prev_wl_1d=wl_state_1d if self.use_flow_first_decoder else None,
                prev_wl_2d=wl_state_2d if self.use_flow_first_decoder else None,
                control_1d=u_1d,
                control_2d=u_2d,
            )
            all_wl_1d.append(decoded['wl_1d'])
            all_wl_2d.append(decoded['wl_2d'])
            all_flow_1d.append(decoded['flow_1d'])
            all_flow_2d.append(decoded['flow_2d'])
            wl_state_1d = decoded['wl_1d']
            wl_state_2d = decoded['wl_2d']

            if t < T - 1:
                if self.use_dual_timescale_latent:
                    z_fast_1d_next = self.transition_1d(
                        z_1d,
                        u_1d,
                        c_e,
                        edge_index_1d,
                        u_proj=u1d_fast_proj_all[:, t] if u1d_fast_proj_all is not None else None,
                        u_mean=u1d_fast_mean_all[:, t] if u1d_fast_mean_all is not None else None,
                        u_std=u1d_fast_std_all[:, t] if u1d_fast_std_all is not None else None,
                        inv_dst_deg=inv_dst_deg_1d,
                    )
                    z_fast_2d_next = self.transition_2d(
                        z_decode_2d,
                        u_2d,
                        c_e,
                        edge_index_2d,
                        u_proj=u2d_fast_proj_all[:, t] if u2d_fast_proj_all is not None else None,
                        u_mean=u2d_fast_mean_all[:, t] if u2d_fast_mean_all is not None else None,
                        u_std=u2d_fast_std_all[:, t] if u2d_fast_std_all is not None else None,
                        inv_dst_deg=inv_dst_deg_2d,
                    )
                    z_slow_1d_next = self.transition_1d_slow(
                        z_slow_1d,
                        u_1d,
                        c_e,
                        edge_index_1d,
                        u_proj=u1d_slow_proj_all[:, t] if u1d_slow_proj_all is not None else None,
                        u_mean=u1d_slow_mean_all[:, t] if u1d_slow_mean_all is not None else None,
                        u_std=u1d_slow_std_all[:, t] if u1d_slow_std_all is not None else None,
                        inv_dst_deg=inv_dst_deg_1d,
                    )
                    z_slow_2d_next = self.transition_2d_slow(
                        z_slow_2d,
                        u_2d,
                        c_e,
                        edge_index_2d,
                        u_proj=u2d_slow_proj_all[:, t] if u2d_slow_proj_all is not None else None,
                        u_mean=u2d_slow_mean_all[:, t] if u2d_slow_mean_all is not None else None,
                        u_std=u2d_slow_std_all[:, t] if u2d_slow_std_all is not None else None,
                        inv_dst_deg=inv_dst_deg_2d,
                    )
                    z_fast_1d_next, z_fast_2d_next = self._apply_cross_coupling(
                        z_fast_1d_next,
                        z_fast_2d_next,
                        src_1d,
                        dst_2d,
                        deg_1d,
                        deg_2d,
                        idx_1d,
                        idx_2d,
                    )
                    z_fast_1d = z_fast_1d_next
                    z_fast_2d = z_fast_2d_next
                    z_slow_1d = z_slow_1d_next
                    z_slow_2d = z_slow_2d_next
                    z_1d = self._fuse_latent(z_fast_1d, z_slow_1d, kind='1d')
                    z_2d = self._fuse_latent(z_fast_2d, z_slow_2d, kind='2d')
                else:
                    z_1d_next = self.transition_1d(
                        z_1d,
                        u_1d,
                        c_e,
                        edge_index_1d,
                        u_proj=u1d_fast_proj_all[:, t] if u1d_fast_proj_all is not None else None,
                        u_mean=u1d_fast_mean_all[:, t] if u1d_fast_mean_all is not None else None,
                        u_std=u1d_fast_std_all[:, t] if u1d_fast_std_all is not None else None,
                        inv_dst_deg=inv_dst_deg_1d,
                    )
                    z_2d_next = self.transition_2d(
                        z_decode_2d,
                        u_2d,
                        c_e,
                        edge_index_2d,
                        u_proj=u2d_fast_proj_all[:, t] if u2d_fast_proj_all is not None else None,
                        u_mean=u2d_fast_mean_all[:, t] if u2d_fast_mean_all is not None else None,
                        u_std=u2d_fast_std_all[:, t] if u2d_fast_std_all is not None else None,
                        inv_dst_deg=inv_dst_deg_2d,
                    )
                    z_1d, z_2d = self._apply_cross_coupling(
                        z_1d_next,
                        z_2d_next,
                        src_1d,
                        dst_2d,
                        deg_1d,
                        deg_2d,
                        idx_1d,
                        idx_2d,
                    )

        # Stack outputs
        pred_wl_1d_ar = torch.stack(all_wl_1d, dim=1)
        pred_wl_2d_ar = torch.stack(all_wl_2d, dim=1)
        pred_wl_1d = pred_wl_1d_ar
        pred_wl_2d = pred_wl_2d_ar

        direct_wl_1d = None
        direct_wl_2d = None
        if self.use_direct_ar_hybrid:
            direct_wl_1d, direct_wl_2d = self._build_direct_predictions(z_0_1d, z_0_2d, c_e, T)
            blend = torch.sigmoid(self.direct_ar_blend_logit)
            pred_wl_1d = blend * pred_wl_1d_ar + (1.0 - blend) * direct_wl_1d
            pred_wl_2d = blend * pred_wl_2d_ar + (1.0 - blend) * direct_wl_2d

        outputs = {
            'pred_wl_1d': pred_wl_1d,  # [B, T, N1, 1]
            'pred_wl_2d': pred_wl_2d,  # [B, T, N2, 1]
            'pred_wl_1d_ar': pred_wl_1d_ar,
            'pred_wl_2d_ar': pred_wl_2d_ar,
            'pred_flow_1d': torch.stack(all_flow_1d, dim=1),  # [B, T, E1, 1]
            'pred_flow_2d': torch.stack(all_flow_2d, dim=1),  # [B, T, E2, 1]
            # KL divergence terms for VAE loss
            'mu_ce': mu_ce,
            'logvar_ce': logvar_ce,
            'mu_z0_1d': mu_z0_1d,
            'logvar_z0_1d': logvar_z0_1d,
            'mu_z0_2d': mu_z0_2d,
            'logvar_z0_2d': logvar_z0_2d,
        }
        if direct_wl_1d is not None and direct_wl_2d is not None:
            outputs['pred_wl_1d_direct'] = direct_wl_1d
            outputs['pred_wl_2d_direct'] = direct_wl_2d
            outputs['direct_ar_blend'] = torch.sigmoid(self.direct_ar_blend_logit)
        if all_inlet_pred_1d is not None:
            outputs['pred_inlet_1d'] = torch.stack(all_inlet_pred_1d, dim=1)

        return outputs


# ==============================================================================
# TRAINING MODULE
# ==============================================================================

class DualFloodTrainer(pl.LightningModule):
    """
    Training module with joint node, edge, and VAE supervision.

    Loss = L_node + λ_edge * L_edge + λ_physics * L_physics + β_ce * KL_ce + β_z0 * KL_z0

    Where:
    - L_node: MSE on water level predictions
    - L_edge: MSE on flow predictions (SUPERVISED!)
    - L_physics: Temporal smoothness / mass conservation
    - KL_ce: KL divergence for event latent c_e
    - KL_z0: KL divergence for initial temporal latent z_0
    """

    def __init__(
        self,
        model: DualFloodGNN,
        graph: HeteroData,
        norm_stats: Dict,
        static_norm_stats: Optional[Dict] = None,
        lr: float = 1e-3,
        lambda_edge: float = 0.1,
        lambda_physics: float = 0.01,
        beta_ce: float = 0.01,  # KL weight for event latent
        beta_z0: float = 0.001,  # KL weight for temporal latent
        edge_loss_type: str = 'huber',
        use_physics_loss: bool = True,
        physics_mode: str = 'smoothness',
        horizon_weight_power: float = 0.0,
        horizon_weight_power_1d: Optional[float] = None,
        horizon_weight_power_2d: Optional[float] = None,
        rollout_len: int = 90,  # Current curriculum rollout length
        lambda_direct_consistency: float = 0.0,
        lambda_inlet: float = 0.0,
        inlet_missing_boost: float = 2.0,
        inlet_loss_warmup_epochs: int = 1,
        architecture_flags: Optional[Dict] = None,
        use_fused_optimizer: bool = True,
        adam_eps: float = 1e-6,
        scheduler_t_max: int = 0,
        scheduler_eta_min: float = 1e-6,
        strict_finite_checks: bool = True,
        finite_check_interval_steps: int = 50,
    ):
        super().__init__()
        self.model = model
        self.graph = graph
        self.norm_stats = norm_stats
        self.static_norm_stats = static_norm_stats
        self.lr = lr
        self.lambda_edge = lambda_edge
        self.lambda_physics = lambda_physics
        self.beta_ce = beta_ce
        self.beta_z0 = beta_z0
        self.edge_loss_type = edge_loss_type
        self.use_physics_loss = use_physics_loss
        self.physics_mode = physics_mode
        self.horizon_weight_power = horizon_weight_power
        self.horizon_weight_power_1d = (
            float(horizon_weight_power) if horizon_weight_power_1d is None
            else float(horizon_weight_power_1d)
        )
        self.horizon_weight_power_2d = (
            float(horizon_weight_power) if horizon_weight_power_2d is None
            else float(horizon_weight_power_2d)
        )
        self.rollout_len = rollout_len
        self.lambda_direct_consistency = lambda_direct_consistency
        self.lambda_inlet = float(lambda_inlet)
        self.inlet_missing_boost = float(max(1.0, inlet_missing_boost))
        self.inlet_loss_warmup_epochs = int(max(0, inlet_loss_warmup_epochs))
        self.architecture_flags = architecture_flags or {}
        self.use_fused_optimizer = use_fused_optimizer
        self.adam_eps = adam_eps
        self.scheduler_t_max = int(scheduler_t_max)
        self.scheduler_eta_min = float(scheduler_eta_min)
        self.strict_finite_checks = strict_finite_checks
        self.finite_check_interval_steps = max(1, int(finite_check_interval_steps))

        self.save_hyperparameters(ignore=['model', 'graph'])

    def forward(self, batch, rollout_len: Optional[int] = None):
        return self.model(
            self.graph,
            batch['input_1d'],
            batch['input_2d'],
            batch['future_rainfall'],
            batch['future_inlet'],
            batch['future_inlet_mask'],
            rollout_len=rollout_len or self.rollout_len,
        )

    def _compute_kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute KL divergence from N(mu, sigma) to N(0, 1)."""
        # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return kl

    def _assert_finite(self, name: str, tensor: torch.Tensor) -> None:
        if not self.strict_finite_checks:
            return
        finite_mask = torch.isfinite(tensor)
        if bool(finite_mask.all()):
            return
        finite_ratio = float(finite_mask.float().mean().item())
        if bool(finite_mask.any()):
            finite_vals = tensor[finite_mask]
            abs_max = float(finite_vals.abs().max().item())
        else:
            abs_max = float('nan')
        raise RuntimeError(
            f"Non-finite tensor detected: {name} "
            f"(finite_ratio={finite_ratio:.6f}, abs_max_finite={abs_max:.6g})"
        )

    def _should_check_finite_train(self, batch_idx: int) -> bool:
        if not self.strict_finite_checks:
            return False
        if batch_idx == 0:
            return True
        return int(self.global_step) % self.finite_check_interval_steps == 0

    def _should_check_finite_val(self, batch_idx: int) -> bool:
        # Validation checks first batch per epoch; enough to catch roll-out explosions.
        return self.strict_finite_checks and batch_idx == 0

    def _resolve_rollout_len(self, batch, full_horizon: bool = False) -> int:
        target_len = batch.get('target_len')
        if torch.is_tensor(target_len):
            max_target_len = int(target_len.max().item())
        elif isinstance(target_len, (list, tuple)) and len(target_len) > 0:
            max_target_len = int(max(target_len))
        else:
            max_target_len = int(batch['target_wl_1d'].shape[1])
        if full_horizon:
            return max(1, max_target_len)
        return max(1, min(int(self.rollout_len), max_target_len))

    def _get_time_mask(
        self,
        batch: Dict[str, torch.Tensor],
        T: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        raw_mask = batch.get('target_mask')
        if raw_mask is None:
            B = batch['target_wl_1d'].shape[0]
            return torch.ones((B, T, 1, 1), device=device, dtype=dtype)
        return raw_mask[:, :T].to(device=device, dtype=dtype)

    @staticmethod
    def _masked_mse(
        pred: torch.Tensor,
        target: torch.Tensor,
        time_mask: torch.Tensor,
        *,
        time_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        weights = time_mask
        if time_weights is not None:
            weights = weights * time_weights
        err2 = (pred - target).pow(2)
        spatial_factor = 1
        for dim in pred.shape[2:]:
            spatial_factor *= int(dim)
        denom = (weights.sum() * spatial_factor).clamp_min(1.0)
        return (err2 * weights).sum() / denom

    @staticmethod
    def _masked_smooth_l1(
        pred: torch.Tensor,
        target: torch.Tensor,
        time_mask: torch.Tensor,
    ) -> torch.Tensor:
        per_elem = F.smooth_l1_loss(pred, target, reduction='none')
        spatial_factor = 1
        for dim in pred.shape[2:]:
            spatial_factor *= int(dim)
        denom = (time_mask.sum() * spatial_factor).clamp_min(1.0)
        return (per_elem * time_mask).sum() / denom

    @staticmethod
    def _build_time_weights(T: int, power: float, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if power <= 0:
            return None
        t = torch.arange(1, T + 1, device=device, dtype=dtype)
        weights = (t / max(T, 1)).pow(power)
        weights = (weights / weights.mean().clamp_min(1e-8)).view(1, T, 1, 1)
        return weights

    def _compute_node_loss(self, outputs, batch):
        """MSE loss on water level predictions."""
        pred_1d = outputs['pred_wl_1d']  # [B, T, N, 1]
        pred_2d = outputs['pred_wl_2d']
        target_1d = batch['target_wl_1d']  # [B, T_full, N, 1]
        target_2d = batch['target_wl_2d']

        # Handle curriculum: only use first rollout_len timesteps of target
        T = pred_1d.shape[1]
        target_1d = target_1d[:, :T]
        target_2d = target_2d[:, :T]
        time_mask = self._get_time_mask(batch, T, device=pred_1d.device, dtype=pred_1d.dtype)

        weights_1d = self._build_time_weights(T, self.horizon_weight_power_1d, pred_1d.device, pred_1d.dtype)
        weights_2d = self._build_time_weights(T, self.horizon_weight_power_2d, pred_2d.device, pred_2d.dtype)
        loss_1d = self._masked_mse(pred_1d, target_1d, time_mask, time_weights=weights_1d)
        loss_2d = self._masked_mse(pred_2d, target_2d, time_mask, time_weights=weights_2d)

        return (loss_1d + loss_2d) / 2

    def _compute_edge_loss(self, outputs, batch):
        """MSE loss on flow predictions - THE KEY SUPERVISED LOSS."""
        pred_flow_1d = outputs['pred_flow_1d']  # [B, T, E, 1]
        pred_flow_2d = outputs['pred_flow_2d']
        target_flow_1d = batch['target_flow_1d']  # [B, T_full, E, 1]
        target_flow_2d = batch['target_flow_2d']

        # Handle curriculum: only use first rollout_len timesteps of target
        T = pred_flow_1d.shape[1]
        target_flow_1d = target_flow_1d[:, :T]
        target_flow_2d = target_flow_2d[:, :T]
        time_mask = self._get_time_mask(batch, T, device=pred_flow_1d.device, dtype=pred_flow_1d.dtype)

        if self.edge_loss_type == 'huber':
            loss_1d = self._masked_smooth_l1(pred_flow_1d, target_flow_1d, time_mask)
            loss_2d = self._masked_smooth_l1(pred_flow_2d, target_flow_2d, time_mask)
        else:
            loss_1d = self._masked_mse(pred_flow_1d, target_flow_1d, time_mask)
            loss_2d = self._masked_mse(pred_flow_2d, target_flow_2d, time_mask)

        return (loss_1d + loss_2d) / 2

    def _compute_direct_ar_consistency(self, outputs, batch) -> torch.Tensor:
        pred_ar_1d = outputs.get('pred_wl_1d_ar')
        pred_ar_2d = outputs.get('pred_wl_2d_ar')
        pred_direct_1d = outputs.get('pred_wl_1d_direct')
        pred_direct_2d = outputs.get('pred_wl_2d_direct')
        if pred_ar_1d is None or pred_ar_2d is None or pred_direct_1d is None or pred_direct_2d is None:
            return torch.tensor(0.0, device=self.device)
        T = pred_ar_1d.shape[1]
        time_mask = self._get_time_mask(batch, T, device=pred_ar_1d.device, dtype=pred_ar_1d.dtype)
        loss_1d = self._masked_smooth_l1(pred_ar_1d, pred_direct_1d, time_mask)
        loss_2d = self._masked_smooth_l1(pred_ar_2d, pred_direct_2d, time_mask)
        return (loss_1d + loss_2d) / 2

    def _compute_inlet_loss(self, outputs, batch) -> torch.Tensor:
        pred_inlet = outputs.get('pred_inlet_1d')
        if pred_inlet is None:
            return torch.tensor(0.0, device=self.device)
        if self.current_epoch < self.inlet_loss_warmup_epochs:
            return torch.tensor(0.0, device=self.device)

        T = pred_inlet.shape[1]
        target_inlet = batch['future_inlet'][:, :T].to(device=pred_inlet.device, dtype=pred_inlet.dtype)
        time_mask = self._get_time_mask(batch, T, device=pred_inlet.device, dtype=pred_inlet.dtype)
        per_elem = F.smooth_l1_loss(pred_inlet, target_inlet, reduction='none')
        base_weights = time_mask.expand_as(per_elem)
        base_loss = (per_elem * base_weights).sum() / base_weights.sum().clamp_min(1.0)

        if self.inlet_missing_boost <= 1.0:
            return base_loss

        inlet_mask = batch.get('future_inlet_mask')
        if inlet_mask is None:
            return base_loss

        missing = (1.0 - inlet_mask[:, :T]).to(device=pred_inlet.device, dtype=pred_inlet.dtype)
        missing_weights = base_weights * missing
        missing_weight_sum = float(missing_weights.sum().item())
        if missing_weight_sum <= 0.0:
            return base_loss
        missing_loss = (per_elem * missing_weights).sum() / missing_weights.sum().clamp_min(1.0)
        return base_loss + (self.inlet_missing_boost - 1.0) * missing_loss

    def _flow_divergence(self, flow: torch.Tensor, edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """
        Compute net inflow-outflow per node.
        Args:
            flow: [B, T, E]
            edge_index: [2, E], source->destination
        Returns:
            divergence: [B, T, N]
        """
        B, T, E = flow.shape
        src = edge_index[0]
        dst = edge_index[1]

        divergence = torch.zeros(B, T, num_nodes, device=flow.device, dtype=flow.dtype)
        src_idx = src.view(1, 1, E).expand(B, T, E)
        dst_idx = dst.view(1, 1, E).expand(B, T, E)
        divergence.scatter_add_(2, dst_idx, flow)   # inflow to destination
        divergence.scatter_add_(2, src_idx, -flow)  # outflow from source
        return divergence

    def _compute_physics_loss(self, outputs, batch):
        """Physics regularization with selectable mode."""
        if not self.use_physics_loss:
            return torch.tensor(0.0, device=self.device)

        pred_wl_1d = outputs['pred_wl_1d']  # [B, T, N, 1]
        pred_wl_2d = outputs['pred_wl_2d']
        T = pred_wl_1d.shape[1]
        time_mask = self._get_time_mask(batch, T, device=pred_wl_1d.device, dtype=pred_wl_1d.dtype)

        if T < 2:
            return torch.tensor(0.0, device=self.device)

        dt_wl_1d = pred_wl_1d[:, 1:] - pred_wl_1d[:, :-1]
        dt_wl_2d = pred_wl_2d[:, 1:] - pred_wl_2d[:, :-1]
        dt_mask = time_mask[:, 1:] * time_mask[:, :-1]
        if float(dt_mask.sum().item()) <= 0.0:
            return torch.tensor(0.0, device=self.device)
        smoothness_loss_1d = self._masked_mse(dt_wl_1d, torch.zeros_like(dt_wl_1d), dt_mask)
        smoothness_loss_2d = self._masked_mse(dt_wl_2d, torch.zeros_like(dt_wl_2d), dt_mask)
        smoothness_loss = (smoothness_loss_1d + smoothness_loss_2d) / 2

        if self.physics_mode == 'smoothness':
            return smoothness_loss

        # Continuity loss: match magnitude of water-level change to magnitude of flow divergence.
        flow_1d = outputs['pred_flow_1d'][..., 0]  # [B, T, E1]
        flow_2d = outputs['pred_flow_2d'][..., 0]  # [B, T, E2]
        wl_1d = outputs['pred_wl_1d'][..., 0]      # [B, T, N1]
        wl_2d = outputs['pred_wl_2d'][..., 0]      # [B, T, N2]

        edge_index_1d = self.graph['1d', 'pipe', '1d'].edge_index.to(self.device)
        edge_index_2d = self.graph['2d', 'surface', '2d'].edge_index.to(self.device)
        n1 = wl_1d.shape[2]
        n2 = wl_2d.shape[2]

        div_1d = self._flow_divergence(flow_1d, edge_index_1d, n1)[:, :-1, :]
        div_2d = self._flow_divergence(flow_2d, edge_index_2d, n2)[:, :-1, :]
        d_wl_1d = wl_1d[:, 1:, :] - wl_1d[:, :-1, :]
        d_wl_2d = wl_2d[:, 1:, :] - wl_2d[:, :-1, :]
        dt_mask_3d = dt_mask.squeeze(-1)  # [B, T-1, 1]

        eps = 1e-4
        def _masked_mean_abs(x: torch.Tensor, mask_3d: torch.Tensor) -> torch.Tensor:
            denom = (mask_3d.sum() * x.shape[2]).clamp_min(1.0)
            return (x.abs() * mask_3d).sum() / denom

        d_wl_1d = d_wl_1d / _masked_mean_abs(d_wl_1d, dt_mask_3d).clamp(min=eps)
        d_wl_2d = d_wl_2d / _masked_mean_abs(d_wl_2d, dt_mask_3d).clamp(min=eps)
        div_1d = div_1d / _masked_mean_abs(div_1d, dt_mask_3d).clamp(min=eps)
        div_2d = div_2d / _masked_mean_abs(div_2d, dt_mask_3d).clamp(min=eps)

        continuity_loss_1d = self._masked_smooth_l1(d_wl_1d.abs(), div_1d.abs(), dt_mask_3d)
        continuity_loss_2d = self._masked_smooth_l1(d_wl_2d.abs(), div_2d.abs(), dt_mask_3d)
        continuity_loss = (continuity_loss_1d + continuity_loss_2d) / 2

        if self.physics_mode == 'continuity':
            return continuity_loss
        if self.physics_mode == 'hybrid':
            return 0.5 * smoothness_loss + 0.5 * continuity_loss
        raise ValueError(f"Invalid physics_mode='{self.physics_mode}'")

    def training_step(self, batch, batch_idx):
        rollout_len = self._resolve_rollout_len(batch, full_horizon=False)
        outputs = self(batch, rollout_len=rollout_len)
        do_check = self._should_check_finite_train(batch_idx)
        if do_check:
            self._assert_finite('pred_wl_1d', outputs['pred_wl_1d'])
            self._assert_finite('pred_wl_2d', outputs['pred_wl_2d'])
            self._assert_finite('pred_flow_1d', outputs['pred_flow_1d'])
            self._assert_finite('pred_flow_2d', outputs['pred_flow_2d'])

        # Compute losses
        loss_node = self._compute_node_loss(outputs, batch)
        loss_edge = self._compute_edge_loss(outputs, batch)
        loss_physics = self._compute_physics_loss(outputs, batch)
        loss_consistency = self._compute_direct_ar_consistency(outputs, batch)
        loss_inlet = self._compute_inlet_loss(outputs, batch)
        if do_check:
            self._assert_finite('loss_node', loss_node)
            self._assert_finite('loss_edge', loss_edge)
            self._assert_finite('loss_physics', loss_physics)
            self._assert_finite('loss_consistency', loss_consistency)
            self._assert_finite('loss_inlet', loss_inlet)

        # KL losses for VAE latents
        kl_ce = self._compute_kl_loss(outputs['mu_ce'], outputs['logvar_ce'])
        kl_z0_1d = self._compute_kl_loss(outputs['mu_z0_1d'], outputs['logvar_z0_1d'])
        kl_z0_2d = self._compute_kl_loss(outputs['mu_z0_2d'], outputs['logvar_z0_2d'])
        kl_z0 = (kl_z0_1d + kl_z0_2d) / 2
        if do_check:
            self._assert_finite('kl_ce', kl_ce)
            self._assert_finite('kl_z0', kl_z0)

        # Total loss
        total_loss = (
            loss_node
            + self.lambda_edge * loss_edge
            + self.lambda_physics * loss_physics
            + self.beta_ce * kl_ce
            + self.beta_z0 * kl_z0
            + self.lambda_direct_consistency * loss_consistency
            + self.lambda_inlet * loss_inlet
        )
        if do_check:
            self._assert_finite('total_loss', total_loss)

        # Log
        self.log('train/loss', total_loss, prog_bar=True)
        self.log('train/loss_node', loss_node)
        self.log('train/loss_edge', loss_edge)
        self.log('train/loss_physics', loss_physics)
        self.log('train/loss_consistency', loss_consistency)
        self.log('train/loss_inlet', loss_inlet)
        self.log('train/kl_ce', kl_ce)
        self.log('train/kl_z0', kl_z0)
        self.log('train/rollout_len', float(rollout_len))
        if 'direct_ar_blend' in outputs:
            self.log('train/direct_ar_blend', outputs['direct_ar_blend'])

        return total_loss

    def validation_step(self, batch, batch_idx):
        # Always validate on full horizon for a stable, comparable checkpoint metric
        # across curriculum stages.
        full_horizon = self._resolve_rollout_len(batch, full_horizon=True)
        outputs = self(batch, rollout_len=full_horizon)
        do_check = self._should_check_finite_val(batch_idx)
        if do_check:
            self._assert_finite('val_pred_wl_1d', outputs['pred_wl_1d'])
            self._assert_finite('val_pred_wl_2d', outputs['pred_wl_2d'])
            self._assert_finite('val_pred_flow_1d', outputs['pred_flow_1d'])
            self._assert_finite('val_pred_flow_2d', outputs['pred_flow_2d'])

        # Compute losses
        loss_node = self._compute_node_loss(outputs, batch)
        loss_edge = self._compute_edge_loss(outputs, batch)
        loss_consistency = self._compute_direct_ar_consistency(outputs, batch)
        loss_inlet = self._compute_inlet_loss(outputs, batch)
        if do_check:
            self._assert_finite('val_loss_node', loss_node)
            self._assert_finite('val_loss_edge', loss_edge)
            self._assert_finite('val_loss_consistency', loss_consistency)
            self._assert_finite('val_loss_inlet', loss_inlet)

        # Compute RMSE in original scale
        pred_1d = outputs['pred_wl_1d']
        pred_2d = outputs['pred_wl_2d']
        target_1d = batch['target_wl_1d']
        target_2d = batch['target_wl_2d']

        # Handle curriculum: slice targets to match prediction length
        T = pred_1d.shape[1]
        target_1d = target_1d[:, :T]
        target_2d = target_2d[:, :T]
        time_mask = self._get_time_mask(batch, T, device=pred_1d.device, dtype=pred_1d.dtype)

        # Denormalize
        std_1d = self.norm_stats['water_level_1d']['std']
        std_2d = self.norm_stats['water_level_2d']['std']
        mse_1d = self._masked_mse(pred_1d, target_1d, time_mask)
        mse_2d = self._masked_mse(pred_2d, target_2d, time_mask)
        rmse_norm_1d = torch.sqrt(mse_1d.clamp_min(1e-12))
        rmse_norm_2d = torch.sqrt(mse_2d.clamp_min(1e-12))
        rmse_1d = rmse_norm_1d * std_1d
        rmse_2d = rmse_norm_2d * std_2d

        # Standardized RMSE (as used in competition)
        std_rmse = (rmse_norm_1d + rmse_norm_2d) / 2
        if do_check:
            self._assert_finite('val_rmse_1d', rmse_1d)
            self._assert_finite('val_rmse_2d', rmse_2d)
            self._assert_finite('val_std_rmse', std_rmse)

        self.log('val/loss_node', loss_node)
        self.log('val/loss_edge', loss_edge)
        self.log('val/loss_consistency', loss_consistency)
        self.log('val/loss_inlet', loss_inlet)
        self.log('val/rmse_1d', rmse_1d)
        self.log('val/rmse_2d', rmse_2d)
        self.log('val/std_rmse', std_rmse)
        self.log('val/std_rmse_full', std_rmse, prog_bar=True)
        if 'direct_ar_blend' in outputs:
            self.log('val/direct_ar_blend', outputs['direct_ar_blend'])

        return {
            'val_loss': (
                loss_node
                + self.lambda_edge * loss_edge
                + self.lambda_direct_consistency * loss_consistency
                + self.lambda_inlet * loss_inlet
            )
        }

    def configure_optimizers(self):
        optimizer_kwargs = {"lr": self.lr, "weight_decay": 1e-5, "eps": self.adam_eps}
        optimizer = None
        clip_val = 0.0
        if self.trainer is not None and self.trainer.gradient_clip_val is not None:
            clip_val = float(self.trainer.gradient_clip_val)
        allow_fused = self.use_fused_optimizer and torch.cuda.is_available() and clip_val <= 0.0

        if self.use_fused_optimizer and clip_val > 0.0:
            print(f"  Fused AdamW disabled because gradient_clip_val={clip_val} > 0")

        if allow_fused:
            try:
                optimizer = torch.optim.AdamW(self.parameters(), fused=True, **optimizer_kwargs)
                print("  Using fused AdamW optimizer")
            except (TypeError, RuntimeError):
                optimizer = None
        if optimizer is None:
            optimizer = torch.optim.AdamW(self.parameters(), **optimizer_kwargs)
        if self.scheduler_t_max > 0:
            scheduler_t_max = max(1, self.scheduler_t_max)
        elif self.trainer is not None and getattr(self.trainer, 'max_epochs', None):
            scheduler_t_max = max(1, int(self.trainer.max_epochs))
        else:
            scheduler_t_max = 30
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=scheduler_t_max, eta_min=self.scheduler_eta_min
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)


# ==============================================================================
# CURRICULUM CALLBACK
# ==============================================================================

class CurriculumCallback(Callback):
    """Callback to advance curriculum stages during training."""

    def __init__(self, stages: List[int], epochs_per_stage: int):
        super().__init__()
        self.stages = stages
        self.epochs_per_stage = epochs_per_stage
        self.current_stage = 0

    def on_train_epoch_start(self, trainer, pl_module):
        # Calculate which stage we should be in
        epoch = trainer.current_epoch
        target_stage = min(epoch // self.epochs_per_stage, len(self.stages) - 1)

        if target_stage > self.current_stage:
            self.current_stage = target_stage
            new_rollout = self.stages[self.current_stage]
            pl_module.rollout_len = new_rollout
            print(f"\n=== Curriculum: Stage {self.current_stage+1}/{len(self.stages)}, rollout_len={new_rollout} ===\n")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description='DualFlood v2: Latent-Space Flood Prediction')
    parser.add_argument('--model_id', type=int, default=2, help='Model ID (1 or 2)')
    parser.add_argument('--data_dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--exp_name', type=str, default='dual_flood_v2', help='Experiment name')
    parser.add_argument('--max_epochs', type=int, default=50, help='Max epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension')
    parser.add_argument('--latent_dim', type=int, default=32, help='Latent dimension')
    parser.add_argument('--num_gnn_layers', type=int, default=3, help='Number of GNN layers')
    parser.add_argument('--transition_scale', type=float, default=0.05,
                        help='Residual step size for latent transition updates')
    parser.add_argument('--coupling_scale', type=float, default=0.05,
                        help='Coupling step size for 1D-2D latent exchange')
    parser.add_argument('--use_flow_first_decoder', action='store_true',
                        help='Enable flow-first mass-conserving water-level decoding')
    parser.add_argument('--use_nodewise_1d_dynamics', action='store_true',
                        help='Enable node-wise 1D mass/decay adaptations in flow-first decoder')
    parser.add_argument('--enable_stable_flow_rollout', action='store_true',
                        help='Enable bounded+dissipative flow-first rollout update')
    parser.add_argument('--use_multiscale_2d', action='store_true',
                        help='Enable multi-scale 2D latent encoder')
    parser.add_argument('--multiscale_num_clusters', type=int, default=128,
                        help='Approximate number of coarse clusters for 2D multi-scale encoder')
    parser.add_argument('--use_moe_transition', action='store_true',
                        help='Use regime-conditioned MoE for latent transitions')
    parser.add_argument('--moe_num_experts', type=int, default=4,
                        help='Number of experts in transition MoE')
    parser.add_argument('--moe_mode', type=str, default='dense', choices=['dense', 'topk'],
                        help="MoE routing mode: dense uses all experts, topk routes to top-k experts")
    parser.add_argument('--moe_top_k', type=int, default=1,
                        help='Top-k experts per sample when moe_mode=topk')
    parser.add_argument('--use_dual_timescale_latent', action='store_true',
                        help='Split latent dynamics into fast and slow states')
    parser.add_argument('--slow_timescale_ratio', type=float, default=0.25,
                        help='Step-size ratio for slow latent transition')
    parser.add_argument('--use_direct_ar_hybrid', action='store_true',
                        help='Blend direct horizon and autoregressive predictions')
    parser.add_argument('--direct_ar_init_blend', type=float, default=0.5,
                        help='Initial AR blend ratio when hybrid head is enabled')
    parser.add_argument('--use_inlet_imputer', action='store_true',
                        help='Infer missing future inlet from latent dynamics and previous inlet')
    parser.add_argument('--disable_precompute_transition_controls', action='store_true',
                        help='Disable rollout-wide precompute of transition control projections')
    parser.add_argument('--disable_precompute_decoder_edge_terms', action='store_true',
                        help='Disable rollout-wide precompute of static edge terms in flow decoder')
    parser.add_argument('--lambda_edge', type=float, default=0.1, help='Edge loss weight')
    parser.add_argument('--lambda_physics', type=float, default=0.01, help='Physics loss weight')
    parser.add_argument('--lambda_direct_consistency', type=float, default=0.0,
                        help='Consistency weight between direct and AR heads')
    parser.add_argument('--lambda_inlet', type=float, default=0.0,
                        help='Auxiliary loss weight for latent-conditioned inlet imputer')
    parser.add_argument('--inlet_missing_boost', type=float, default=2.0,
                        help='Additional weighting for inlet-imputer loss on masked/missing timesteps')
    parser.add_argument('--inlet_loss_warmup_epochs', type=int, default=1,
                        help='Disable inlet-imputer supervision for first N epochs')
    parser.add_argument(
        '--physics_mode',
        type=str,
        default='smoothness',
        choices=['smoothness', 'continuity', 'hybrid'],
        help='Physics regularization mode'
    )
    parser.add_argument(
        '--horizon_weight_power',
        type=float,
        default=0.0,
        help='If >0, upweight long horizons in node loss with ((t/T)^power)'
    )
    parser.add_argument(
        '--horizon_weight_power_1d',
        type=float,
        default=None,
        help='Optional 1D-specific horizon weighting power (default: use --horizon_weight_power)'
    )
    parser.add_argument(
        '--horizon_weight_power_2d',
        type=float,
        default=None,
        help='Optional 2D-specific horizon weighting power (default: use --horizon_weight_power)'
    )
    parser.add_argument('--edge_loss_type', type=str, default='huber', choices=['mse', 'huber'],
                        help='Loss type for supervised edge flow training')
    parser.add_argument('--beta_ce', type=float, default=0.01, help='KL weight for event latent')
    parser.add_argument('--beta_z0', type=float, default=0.001, help='KL weight for temporal latent')
    parser.add_argument('--accelerator', type=str, default='auto', help='Accelerator')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--pred_len', type=int, default=90, help='Prediction length')
    parser.add_argument('--min_pred_len', type=int, default=1,
                        help='Minimum supervised future length per sample (supports variable horizons)')
    parser.add_argument('--horizon_sampling_power', type=float, default=0.0,
                        help='If >0, sample long-horizon windows more often: weight ~ (target_len/pred_len)^power')
    parser.add_argument('--resume_ckpt', type=str, default='',
                        help='Optional checkpoint path to resume/fine-tune from')
    parser.add_argument('--init_ckpt', type=str, default='',
                        help='Optional checkpoint path to initialize model weights only '
                             '(does not restore optimizer/epoch/callback states)')
    parser.add_argument(
        '--freeze_pretrained',
        action='store_true',
        help='When --init_ckpt is used, freeze loaded parameters and train only newly introduced ones'
    )
    parser.add_argument('--accumulate_grad_batches', type=int, default=8, help='Gradient accumulation')
    parser.add_argument('--num_workers', type=int, default=8, help='DataLoader workers')
    parser.add_argument('--prefetch_factor', type=int, default=4, help='DataLoader prefetch factor')
    parser.add_argument('--disable_persistent_workers', action='store_true', help='Disable DataLoader persistent workers')
    parser.add_argument('--disable_pin_memory', action='store_true', help='Disable DataLoader pin_memory')
    parser.add_argument('--drop_last_train', action='store_true', help='Drop last incomplete train batch')
    parser.add_argument('--disable_norm_cache', action='store_true', help='Disable cached normalization stats')
    parser.add_argument('--norm_cache_path', type=str, default='',
                        help='Optional explicit path for normalization stats cache (.npz)')
    parser.add_argument(
        '--precision',
        type=str,
        default='32-true',
        help='Lightning precision (e.g., 32-true, 16-mixed, bf16-mixed)'
    )
    parser.add_argument('--torch_compile', action='store_true', help='Enable torch.compile on model')
    parser.add_argument(
        '--compile_mode',
        type=str,
        default='reduce-overhead',
        choices=['default', 'reduce-overhead', 'max-autotune'],
        help='torch.compile mode'
    )
    parser.add_argument('--disable_tf32', action='store_true', help='Disable TF32 matmul/cuDNN kernels')
    parser.add_argument('--disable_benchmark', action='store_true', help='Disable cuDNN benchmark autotuning')
    parser.add_argument('--disable_fused_optimizer', action='store_true',
                        help='Disable fused AdamW optimizer on CUDA')
    parser.add_argument('--gradient_clip_val', type=float, default=1.0,
                        help='Gradient clipping value (set 0 to disable)')
    parser.add_argument('--adam_eps', type=float, default=1e-6,
                        help='AdamW epsilon for mixed-precision stability')
    parser.add_argument('--lr_scheduler_t_max', type=int, default=0,
                        help='Cosine scheduler T_max in epochs (0 = use max_epochs)')
    parser.add_argument('--lr_scheduler_eta_min', type=float, default=1e-6,
                        help='Cosine scheduler minimum LR')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                        help='Early stopping patience on val/std_rmse_full')
    parser.add_argument('--disable_strict_finite_checks', action='store_true',
                        help='Disable fail-fast checks for non-finite tensors/losses')
    parser.add_argument('--finite_check_interval_steps', type=int, default=50,
                        help='Run strict finite checks every N optimizer steps (and first batch)')
    parser.add_argument('--limit_train_batches', type=float, default=1.0, help='Fraction or count of train batches per epoch')
    parser.add_argument('--limit_val_batches', type=float, default=1.0, help='Fraction or count of validation batches per epoch')
    parser.add_argument(
        '--future_inlet_mode_train',
        type=str,
        default='observed',
        choices=['observed', 'missing', 'mixed', 'zero', 'last'],
        help='How to provide future inlet_flow during training'
    )
    parser.add_argument(
        '--future_inlet_mode_val',
        type=str,
        default='observed',
        choices=['observed', 'missing', 'mixed', 'zero', 'last'],
        help='How to provide future inlet_flow during validation'
    )
    parser.add_argument(
        '--train_start_only',
        action='store_true',
        help='Train only from event start (start=0), aligning train objective with test rollout'
    )
    parser.add_argument(
        '--future_inlet_dropout_prob_train',
        type=float,
        default=0.0,
        help='Element-wise dropout probability for future inlet when mode=mixed (train)'
    )
    parser.add_argument(
        '--future_inlet_seq_dropout_prob_train',
        type=float,
        default=0.0,
        help='Whole-sequence dropout probability for future inlet when mode=mixed (train)'
    )
    parser.add_argument(
        '--future_inlet_dropout_prob_val',
        type=float,
        default=0.0,
        help='Element-wise dropout probability for future inlet when mode=mixed (val)'
    )
    parser.add_argument(
        '--future_inlet_seq_dropout_prob_val',
        type=float,
        default=0.0,
        help='Whole-sequence dropout probability for future inlet when mode=mixed (val)'
    )
    parser.add_argument(
        '--val_start_only',
        action='store_true',
        help='Validate only from event start (start=0) to mimic test-time rollout'
    )
    parser.add_argument('--log_every_n_steps', type=int, default=50, help='Logging frequency in steps')
    parser.add_argument('--enable_progress_bar', action='store_true',
                        help='Enable progress bar output (off by default for faster nohup runs)')
    # Curriculum learning
    parser.add_argument('--use_curriculum', action='store_true', help='Use curriculum learning')
    parser.add_argument('--curriculum_stages', type=str, default='4,8,16,32,64,90',
                        help='Curriculum stages (comma-separated rollout lengths)')
    parser.add_argument('--epochs_per_stage', type=int, default=5, help='Epochs per curriculum stage')

    args = parser.parse_args()

    if args.resume_ckpt and args.init_ckpt:
        raise ValueError("Use either --resume_ckpt or --init_ckpt, not both.")
    if args.min_pred_len < 1 or args.min_pred_len > args.pred_len:
        raise ValueError("--min_pred_len must be in [1, pred_len]")

    # Set seed
    pl.seed_everything(args.seed)

    if torch.cuda.is_available():
        enable_tf32 = not args.disable_tf32
        torch.backends.cuda.matmul.allow_tf32 = enable_tf32
        torch.backends.cudnn.allow_tf32 = enable_tf32
        if enable_tf32:
            torch.set_float32_matmul_precision('high')

    print("=" * 70)
    print(f"DualFlood v2 Training - Model {args.model_id}")
    print(f"Experiment: {args.exp_name}")
    print(f"Key: Latent space + Supervised edge flows")
    print("=" * 70)

    # Scale dimensions for Model 2 (larger network)
    if args.model_id == 2:
        args.hidden_dim = 96
        args.latent_dim = 48
        args.num_gnn_layers = 4

    print(f"Config: hidden={args.hidden_dim}, latent={args.latent_dim}, gnn_layers={args.num_gnn_layers}")
    print(f"Dynamics scales: transition={args.transition_scale}, coupling={args.coupling_scale}")
    print(f"Loss weights: edge={args.lambda_edge}, physics={args.lambda_physics}")
    print(f"Direct/AR consistency weight: {args.lambda_direct_consistency}")
    print(
        f"Inlet imputer loss weight: {args.lambda_inlet} "
        f"(missing boost={args.inlet_missing_boost}, warmup_epochs={args.inlet_loss_warmup_epochs})"
    )
    print(f"Physics mode: {args.physics_mode}")
    print(f"Horizon weight power: {args.horizon_weight_power}")
    if args.horizon_weight_power_1d is not None or args.horizon_weight_power_2d is not None:
        print(
            "Horizon weight split: "
            f"1d={args.horizon_weight_power_1d if args.horizon_weight_power_1d is not None else args.horizon_weight_power}, "
            f"2d={args.horizon_weight_power_2d if args.horizon_weight_power_2d is not None else args.horizon_weight_power}"
        )
    print(f"Prediction horizon: max={args.pred_len}, min_supervised={args.min_pred_len}")
    print(f"Horizon sampling power: {args.horizon_sampling_power}")
    print(f"Edge supervision loss: {args.edge_loss_type}")
    print(f"VAE weights: beta_ce={args.beta_ce}, beta_z0={args.beta_z0}")
    print(
        "Architecture flags: "
        f"flow_first={args.use_flow_first_decoder}, "
        f"multiscale2d={args.use_multiscale_2d}, "
        f"moe_transition={args.use_moe_transition}, "
        f"dual_timescale={args.use_dual_timescale_latent}, "
        f"direct_ar_hybrid={args.use_direct_ar_hybrid}, "
        f"inlet_imputer={args.use_inlet_imputer}, "
        f"nodewise_1d_dynamics={args.use_nodewise_1d_dynamics}, "
        f"stable_flow_rollout={args.enable_stable_flow_rollout}, "
        f"precompute_transition={not args.disable_precompute_transition_controls}, "
        f"precompute_decoder_edges={not args.disable_precompute_decoder_edge_terms}"
    )
    if args.use_multiscale_2d:
        print(f"  Multi-scale clusters: {args.multiscale_num_clusters}")
    if args.use_moe_transition:
        print(f"  Transition experts: {args.moe_num_experts}")
        print(f"  MoE routing: mode={args.moe_mode}, top_k={args.moe_top_k}")
    if args.use_dual_timescale_latent:
        print(f"  Slow timescale ratio: {args.slow_timescale_ratio}")
    if args.use_direct_ar_hybrid:
        print(f"  Initial AR blend: {args.direct_ar_init_blend}")
    print(
        "Future inlet modes: "
        f"train={args.future_inlet_mode_train}, val={args.future_inlet_mode_val}, "
        f"train_start_only={args.train_start_only}, val_start_only={args.val_start_only}"
    )
    print(
        "Future inlet dropout: "
        f"train(elem={args.future_inlet_dropout_prob_train}, seq={args.future_inlet_seq_dropout_prob_train}), "
        f"val(elem={args.future_inlet_dropout_prob_val}, seq={args.future_inlet_seq_dropout_prob_val})"
    )
    print(
        f"Scheduler: cosine(T_max={args.lr_scheduler_t_max or args.max_epochs}, "
        f"eta_min={args.lr_scheduler_eta_min}) | early_stopping_patience={args.early_stopping_patience}"
    )

    # Create data module
    print("\n[1/4] Loading data...")
    print(f"Using pred_len={args.pred_len} (shorter = less memory)")
    data_module = DualFloodDataModule(
        data_dir=args.data_dir,
        model_id=args.model_id,
        batch_size=args.batch_size,
        seq_len=10,
        pred_len=args.pred_len,
        min_pred_len=args.min_pred_len,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        pin_memory=not args.disable_pin_memory,
        persistent_workers=not args.disable_persistent_workers,
        drop_last_train=args.drop_last_train,
        use_norm_cache=not args.disable_norm_cache,
        norm_cache_path=args.norm_cache_path,
        future_inlet_mode_train=args.future_inlet_mode_train,
        future_inlet_mode_val=args.future_inlet_mode_val,
        train_start_only=args.train_start_only,
        val_start_only=args.val_start_only,
        future_inlet_dropout_prob_train=args.future_inlet_dropout_prob_train,
        future_inlet_seq_dropout_prob_train=args.future_inlet_seq_dropout_prob_train,
        future_inlet_dropout_prob_val=args.future_inlet_dropout_prob_val,
        future_inlet_seq_dropout_prob_val=args.future_inlet_seq_dropout_prob_val,
        horizon_sampling_power=args.horizon_sampling_power,
    )
    data_module.setup()

    # Build graph
    print("\n[2/4] Building graph...")
    graph_builder = DualFloodGraphBuilder(args.data_dir, args.model_id)
    static_norm_stats = graph_builder.compute_static_norm_stats()
    graph_builder = DualFloodGraphBuilder(args.data_dir, args.model_id, static_norm_stats=static_norm_stats)
    graph = graph_builder.build(split="train")

    # Get edge dimensions
    edge_dim_1d = graph['1d', 'pipe', '1d'].edge_attr.shape[1] if hasattr(graph['1d', 'pipe', '1d'], 'edge_attr') else 4
    edge_dim_2d = graph['2d', 'surface', '2d'].edge_attr.shape[1] if hasattr(graph['2d', 'surface', '2d'], 'edge_attr') else 4

    # Create model
    print("\n[3/4] Creating model...")
    model = DualFloodGNN(
        num_1d_nodes=data_module.num_1d_nodes,
        num_2d_nodes=data_module.num_2d_nodes,
        num_1d_edges=data_module.num_1d_edges,
        num_2d_edges=data_module.num_2d_edges,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_gnn_layers=args.num_gnn_layers,
        edge_dim_1d=edge_dim_1d,
        edge_dim_2d=edge_dim_2d,
        pred_len=args.pred_len,
        transition_scale=args.transition_scale,
        coupling_scale=args.coupling_scale,
        use_flow_first_decoder=args.use_flow_first_decoder,
        use_multiscale_2d=args.use_multiscale_2d,
        multiscale_num_clusters=args.multiscale_num_clusters,
        use_moe_transition=args.use_moe_transition,
        moe_num_experts=args.moe_num_experts,
        moe_mode=args.moe_mode,
        moe_top_k=args.moe_top_k,
        use_dual_timescale_latent=args.use_dual_timescale_latent,
        slow_timescale_ratio=args.slow_timescale_ratio,
        use_direct_ar_hybrid=args.use_direct_ar_hybrid,
        direct_ar_init_blend=args.direct_ar_init_blend,
        use_inlet_imputer=args.use_inlet_imputer,
        use_nodewise_1d_dynamics=args.use_nodewise_1d_dynamics,
        precompute_transition_controls=not args.disable_precompute_transition_controls,
        precompute_decoder_edge_terms=not args.disable_precompute_decoder_edge_terms,
        use_stable_flow_rollout=args.enable_stable_flow_rollout,
    )

    if args.init_ckpt:
        print(f"Initializing model weights from: {args.init_ckpt}")
        ckpt = torch.load(args.init_ckpt, map_location='cpu', weights_only=False)
        raw_state = ckpt.get('state_dict', ckpt)
        if any(k.startswith('model.') for k in raw_state.keys()):
            model_state = {k[len('model.'):]: v for k, v in raw_state.items() if k.startswith('model.')}
        else:
            model_state = raw_state
        model_state = adapt_init_state_for_new_architecture(
            model_state=model_state,
            use_moe_transition=args.use_moe_transition,
            moe_num_experts=args.moe_num_experts,
            use_dual_timescale_latent=args.use_dual_timescale_latent,
        )
        skipped_keys, missing_keys = load_matching_state_dict(model, model_state)
        if skipped_keys:
            print(f"  Skipped incompatible init_ckpt keys: {len(skipped_keys)} (first 10: {skipped_keys[:10]})")
        if missing_keys:
            print(f"  Missing model keys after init load: {len(missing_keys)} (first 10: {missing_keys[:10]})")

        if args.freeze_pretrained:
            missing_set = set(missing_keys)
            trainable_names = []
            frozen_names = []
            for name, param in model.named_parameters():
                if name in missing_set:
                    param.requires_grad = True
                    trainable_names.append(name)
                else:
                    param.requires_grad = False
                    frozen_names.append(name)

            print(
                f"  freeze_pretrained enabled: trainable_params={len(trainable_names)}, "
                f"frozen_params={len(frozen_names)}"
            )
            if trainable_names:
                print(f"  First trainable params: {trainable_names[:10]}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    if args.torch_compile:
        if hasattr(torch, "compile"):
            try:
                model = torch.compile(model, mode=args.compile_mode)
                print(f"  torch.compile enabled (mode={args.compile_mode})")
            except Exception as exc:
                print(f"  torch.compile failed, continuing without compile: {exc}")
        else:
            print("  torch.compile unavailable in this torch build")

    # Parse curriculum stages
    if args.use_curriculum:
        curriculum_stages = [int(x) for x in args.curriculum_stages.split(',')]
        initial_rollout = curriculum_stages[0]
    else:
        curriculum_stages = [args.pred_len]
        initial_rollout = args.pred_len

    print(f"  Curriculum: {curriculum_stages}")

    # Create trainer module
    architecture_flags = {
        'use_flow_first_decoder': args.use_flow_first_decoder,
        'use_multiscale_2d': args.use_multiscale_2d,
        'multiscale_num_clusters': args.multiscale_num_clusters,
        'use_moe_transition': args.use_moe_transition,
        'moe_num_experts': args.moe_num_experts,
        'moe_mode': args.moe_mode,
        'moe_top_k': args.moe_top_k,
        'use_dual_timescale_latent': args.use_dual_timescale_latent,
        'slow_timescale_ratio': args.slow_timescale_ratio,
        'use_direct_ar_hybrid': args.use_direct_ar_hybrid,
        'use_inlet_imputer': args.use_inlet_imputer,
        'use_nodewise_1d_dynamics': args.use_nodewise_1d_dynamics,
        'direct_ar_init_blend': args.direct_ar_init_blend,
        'use_stable_flow_rollout': args.enable_stable_flow_rollout,
        'precompute_transition_controls': not args.disable_precompute_transition_controls,
        'precompute_decoder_edge_terms': not args.disable_precompute_decoder_edge_terms,
    }
    trainer_module = DualFloodTrainer(
        model=model,
        graph=graph,
        norm_stats=data_module.norm_stats,
        static_norm_stats=static_norm_stats,
        lr=args.lr,
        lambda_edge=args.lambda_edge,
        lambda_physics=args.lambda_physics,
        physics_mode=args.physics_mode,
        horizon_weight_power=args.horizon_weight_power,
        horizon_weight_power_1d=args.horizon_weight_power_1d,
        horizon_weight_power_2d=args.horizon_weight_power_2d,
        beta_ce=args.beta_ce,
        beta_z0=args.beta_z0,
        edge_loss_type=args.edge_loss_type,
        rollout_len=initial_rollout,
        lambda_direct_consistency=args.lambda_direct_consistency,
        lambda_inlet=args.lambda_inlet,
        inlet_missing_boost=args.inlet_missing_boost,
        inlet_loss_warmup_epochs=args.inlet_loss_warmup_epochs,
        architecture_flags=architecture_flags,
        use_fused_optimizer=not args.disable_fused_optimizer,
        adam_eps=args.adam_eps,
        scheduler_t_max=args.lr_scheduler_t_max,
        scheduler_eta_min=args.lr_scheduler_eta_min,
        strict_finite_checks=not args.disable_strict_finite_checks,
        finite_check_interval_steps=args.finite_check_interval_steps,
    )

    # Callbacks
    checkpoint_dir = f"checkpoints/model_{args.model_id}/{args.exp_name}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename='best',
            monitor='val/std_rmse_full',
            mode='min',
            save_top_k=1,
            save_last=True,
        ),
        EarlyStopping(
            monitor='val/std_rmse_full',
            patience=args.early_stopping_patience,
            mode='min',
        ),
    ]

    # Create trainer
    print("\n[4/4] Starting training...")
    print(f"Gradient accumulation: {args.accumulate_grad_batches} batches")
    print(f"DataLoader workers: {args.num_workers}")
    print(f"DataLoader prefetch_factor: {args.prefetch_factor}")
    print(f"Pin memory: {not args.disable_pin_memory}")
    print(f"Persistent workers: {not args.disable_persistent_workers}")
    print(f"Drop last train batch: {args.drop_last_train}")
    print(f"Precision: {args.precision}")
    print(f"TF32 enabled: {not args.disable_tf32}")
    print(f"cuDNN benchmark: {not args.disable_benchmark}")
    print(f"Fused optimizer: {not args.disable_fused_optimizer}")
    print(f"Gradient clip val: {args.gradient_clip_val}")
    print(f"AdamW eps: {args.adam_eps}")
    print(f"Strict finite checks: {not args.disable_strict_finite_checks}")
    print(f"Finite check interval: {args.finite_check_interval_steps} steps")
    print(f"torch.compile: {args.torch_compile} (mode={args.compile_mode})")
    if args.resume_ckpt:
        print(f"Resuming from checkpoint: {args.resume_ckpt}")

    # Add curriculum callback if enabled
    if args.use_curriculum and len(curriculum_stages) > 1:
        print(f"Using curriculum learning: {curriculum_stages}")
        callbacks.append(CurriculumCallback(curriculum_stages, args.epochs_per_stage))

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        callbacks=callbacks,
        num_sanity_val_steps=0,
        enable_progress_bar=args.enable_progress_bar,
        log_every_n_steps=args.log_every_n_steps,
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        precision=args.precision,
        benchmark=not args.disable_benchmark,
        limit_train_batches=args.limit_train_batches,
        limit_val_batches=args.limit_val_batches,
    )
    fit_kwargs = {}
    if args.resume_ckpt:
        fit_kwargs['ckpt_path'] = args.resume_ckpt
    trainer.fit(trainer_module, data_module, **fit_kwargs)

    print("\n" + "=" * 70)
    print("Training completed!")
    print(f"Best checkpoint: {checkpoint_dir}/best.ckpt")
    print("=" * 70)


if __name__ == '__main__':
    main()
