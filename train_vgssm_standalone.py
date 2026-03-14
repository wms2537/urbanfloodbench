#!/usr/bin/env python3
"""
VGSSM: Variational Graph State-Space Model - Standalone Training Script
Physics-Informed Version with Conservation Losses and Curriculum Learning

This single file contains everything needed to train Physics-Informed VGSSM:
- Model architecture (LatentTransition, LatentInferenceNet, LatentDecoder, VGSSM)
- Physics components (EdgeFlowHead, MassConservationLoss)
- Curriculum learning scheduler
- Training utilities (VGSSMTrainer)
- Data loading
- Training loop

Key Physics-Informed Features:
- Delta prediction: h_{t+1} = h_t + delta_h (more stable for long rollouts)
- Edge flow prediction: Q_ij on each edge type (pipe, surface, coupling)
- Local mass conservation: dV_i/dt = In_i - Out_i + S_i
- Global mass conservation: sum(dV/dt) = In_global - Out_global + S_global
- Curriculum learning: Progressive rollout 1→4→8→16→32→90 timesteps

Usage:
    # Standard training
    python train_vgssm_standalone.py --model_id 1 --exp_name vgssm_physics --max_epochs 50

    # With physics losses and curriculum
    python train_vgssm_standalone.py --model_id 1 --exp_name vgssm_physics \
        --use_physics_loss --use_delta_prediction --use_curriculum --max_epochs 50
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import math
import re

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor, Callback
from pytorch_lightning.loggers import TensorBoardLogger

from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, GATConv

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.data.graph_builder import FloodGraphBuilder


# =============================================================================
# COUPLED GNN (from coupled_gnn.py)
# =============================================================================

class HeteroGraphConv(nn.Module):
    """Single heterogeneous graph convolution layer."""

    def __init__(
        self,
        in_channels_dict: Dict[str, int],
        out_channels: int,
        edge_types: List[Tuple[str, str, str]],
        aggr: str = 'mean',
        use_attention: bool = False,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.use_attention = use_attention

        conv_dict = {}
        for src_type, edge_name, dst_type in edge_types:
            in_channels = (in_channels_dict[src_type], in_channels_dict[dst_type])
            if use_attention:
                conv = GATConv(
                    in_channels,
                    out_channels // heads,
                    heads=heads,
                    concat=True,
                    dropout=dropout,
                    add_self_loops=False,
                )
            else:
                conv = SAGEConv(in_channels, out_channels, aggr=aggr)
            conv_dict[(src_type, edge_name, dst_type)] = conv

        self.conv = HeteroConv(conv_dict, aggr='sum')
        self.norms = nn.ModuleDict({
            node_type: nn.LayerNorm(out_channels)
            for node_type in in_channels_dict.keys()
        })
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_dict, edge_index_dict):
        out_dict = self.conv(x_dict, edge_index_dict)
        for node_type, x in out_dict.items():
            x = self.norms[node_type](x)
            x = F.gelu(x)
            x = self.dropout(x)
            out_dict[node_type] = x
        return out_dict


class CoupledHeteroGNN(nn.Module):
    """Multi-layer Coupled Heterogeneous GNN for 1D-2D flood networks."""

    def __init__(
        self,
        in_channels_1d: int,
        in_channels_2d: int,
        hidden_channels: int = 64,
        out_channels: int = 64,
        num_layers: int = 3,
        use_attention: bool = True,
        heads: int = 4,
        dropout: float = 0.1,
        residual: bool = True,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.residual = residual

        self.edge_types = [
            ('1d', 'pipe', '1d'),
            ('2d', 'surface', '2d'),
            ('1d', 'couples_to', '2d'),
            ('2d', 'couples_from', '1d'),
        ]

        self.input_proj = nn.ModuleDict({
            '1d': nn.Linear(in_channels_1d, hidden_channels),
            '2d': nn.Linear(in_channels_2d, hidden_channels),
        })

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            in_ch = hidden_channels
            out_ch = hidden_channels if i < num_layers - 1 else out_channels
            self.convs.append(
                HeteroGraphConv(
                    in_channels_dict={'1d': in_ch, '2d': in_ch},
                    out_channels=out_ch,
                    edge_types=self.edge_types,
                    use_attention=use_attention,
                    heads=heads,
                    dropout=dropout,
                )
            )

        if residual and hidden_channels != out_channels:
            self.res_proj = nn.ModuleDict({
                '1d': nn.Linear(hidden_channels, out_channels),
                '2d': nn.Linear(hidden_channels, out_channels),
            })
        else:
            self.res_proj = None

    def forward(self, x_dict, edge_index_dict):
        h_dict = {
            node_type: self.input_proj[node_type](x)
            for node_type, x in x_dict.items()
        }

        for i, conv in enumerate(self.convs):
            h_prev = h_dict
            h_dict = conv(h_dict, edge_index_dict)
            if self.residual and i < self.num_layers - 1:
                for node_type in h_dict:
                    h_dict[node_type] = h_dict[node_type] + h_prev[node_type]

        if self.residual and self.res_proj is not None:
            for node_type in h_dict:
                h_dict[node_type] = h_dict[node_type] + self.res_proj[node_type](h_prev[node_type])

        return h_dict

    def forward_from_data(self, data: HeteroData):
        x_dict = {'1d': data['1d'].x, '2d': data['2d'].x}
        edge_index_dict = {
            edge_type: data[edge_type].edge_index
            for edge_type in self.edge_types
            if edge_type in data.edge_types
        }
        return self(x_dict, edge_index_dict)


class SpatialEncoder(nn.Module):
    """Encodes spatial/static features using the coupled GNN."""

    def __init__(self, static_1d_dim, static_2d_dim, hidden_channels=64, num_layers=2, **kwargs):
        super().__init__()
        self.gnn = CoupledHeteroGNN(
            in_channels_1d=static_1d_dim,
            in_channels_2d=static_2d_dim,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=num_layers,
            **kwargs
        )

    def forward(self, data: HeteroData):
        out = self.gnn.forward_from_data(data)
        return out['1d'], out['2d']


# =============================================================================
# TFT COMPONENTS (from tft.py)
# =============================================================================

class GatedLinearUnit(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim * 2)

    def forward(self, x):
        out = self.linear(x)
        gate = torch.sigmoid(out[..., :out.shape[-1]//2])
        return out[..., out.shape[-1]//2:] * gate


class GatedResidualNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, context_dim=None, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        if context_dim is not None:
            self.context_proj = nn.Linear(context_dim, hidden_dim, bias=False)
        else:
            self.context_proj = None

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.gate = GatedLinearUnit(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)

        if input_dim != output_dim:
            self.skip_proj = nn.Linear(input_dim, output_dim)
        else:
            self.skip_proj = None

    def forward(self, x, context=None):
        hidden = F.elu(self.fc1(x))
        if context is not None and self.context_proj is not None:
            hidden = hidden + self.context_proj(context)
        hidden = F.elu(self.fc2(hidden))
        hidden = self.dropout(hidden)
        gated = self.gate(hidden)
        if self.skip_proj is not None:
            skip = self.skip_proj(x)
        else:
            skip = x
        return self.layer_norm(gated + skip)


# =============================================================================
# EVENT LATENT ENCODER (from graph_tft.py)
# =============================================================================

class EventLatentEncoderTFT(nn.Module):
    """Event latent encoder using attention for better event representation."""

    def __init__(self, input_dim, hidden_dim, latent_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.pool_proj = GatedResidualNetwork(hidden_dim, hidden_dim, hidden_dim, dropout=dropout)
        self.mean_proj = nn.Linear(hidden_dim, latent_dim)
        self.logvar_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, mask=None):
        batch, seq_len, num_nodes, input_dim = x.shape
        x_pooled = x.mean(dim=2)
        h = self.input_proj(x_pooled)
        h_attn, _ = self.attention(h, h, h)
        h = h + h_attn
        pooled = h.mean(dim=1)
        pooled = self.pool_proj(pooled)
        mean = self.mean_proj(pooled)
        logvar = self.logvar_proj(pooled)
        logvar = torch.clamp(logvar, min=-10, max=2)
        return mean, logvar

    def sample(self, mean, logvar, temperature=1.0):
        std = torch.exp(0.5 * logvar) * temperature
        eps = torch.randn_like(std)
        return mean + eps * std


# =============================================================================
# PHYSICS-INFORMED COMPONENTS
# =============================================================================

class EdgeFlowHead(nn.Module):
    """DEPRECATED: Old approach that predicts flow from latent states.

    This is kept for backward compatibility but should not be used.
    Use PhysicsBasedFlow instead which computes flow from water level differences.
    """

    def __init__(self, latent_dim: int, hidden_dim: int, edge_attr_dim: int = 0,
                 dropout: float = 0.1, max_flow: float = 100.0):
        super().__init__()
        input_dim = latent_dim * 2 + edge_attr_dim
        self.max_flow = max_flow

        self.flow_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        self.direction_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh(),
        )

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor,
                edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        if edge_attr is not None:
            combined = torch.cat([z_src, z_dst, edge_attr], dim=-1)
        else:
            combined = torch.cat([z_src, z_dst], dim=-1)

        raw_magnitude = self.flow_net(combined)
        magnitude = self.max_flow * torch.sigmoid(raw_magnitude)
        direction = self.direction_net(combined)
        flow = magnitude * direction
        return flow


class PhysicsBasedFlow(nn.Module):
    """Computes flow from WATER LEVEL differences using physical equations.

    This is the proper physics-informed approach where:
    - Flow Q_ij depends on head difference (h_i - h_j)
    - Uses simplified Manning's/weir equations
    - Learnable conductance parameters per edge type

    Key insight: Flow should be DERIVED from water levels, not predicted independently.
    This ensures the conservation equation is physically meaningful.

    Physics equations used:
    - For pipes (1D): Q = K * sign(Δh) * |Δh|^0.5  (orifice-like)
    - For surface (2D): Q = K * Δh  (linear approximation for shallow flow)
    - For coupling: Q = K * sign(Δh) * |Δh|^0.5  (weir-like exchange)

    Where K is a learnable conductance that depends on geometry.
    """

    def __init__(self, hidden_dim: int = 64, max_flow: float = 10.0):
        super().__init__()
        self.max_flow = max_flow

        # Learnable conductance for each edge type
        # These represent hydraulic conductivity/permeability
        # Initialized to reasonable values for water flow
        self.log_K_pipe = nn.Parameter(torch.tensor(0.0))  # K ~ 1.0 initially
        self.log_K_surface = nn.Parameter(torch.tensor(-1.0))  # K ~ 0.37 (slower surface flow)
        self.log_K_coupling = nn.Parameter(torch.tensor(-0.5))  # K ~ 0.6 (exchange)

        # Optional: learnable exponent for head-flow relationship
        # Default 0.5 is for orifice/weir flow, 1.0 is Darcy flow
        self.alpha_pipe = nn.Parameter(torch.tensor(0.5))
        self.alpha_surface = nn.Parameter(torch.tensor(1.0))  # Linear for shallow 2D
        self.alpha_coupling = nn.Parameter(torch.tensor(0.5))

    def forward(self, h_src: torch.Tensor, h_dst: torch.Tensor,
                edge_type: str) -> torch.Tensor:
        """
        Compute flow from source to destination based on water level difference.

        Args:
            h_src: Water level at source nodes [num_edges]
            h_dst: Water level at destination nodes [num_edges]
            edge_type: 'pipe', 'surface', or 'coupling'

        Returns:
            flow: Signed flow [num_edges], positive = src->dst
        """
        # Head difference drives flow
        delta_h = h_src - h_dst  # [num_edges]

        # Select parameters based on edge type
        if edge_type == 'pipe':
            K = torch.exp(self.log_K_pipe)
            alpha = torch.sigmoid(self.alpha_pipe) + 0.3  # Range [0.3, 1.3]
        elif edge_type == 'surface':
            K = torch.exp(self.log_K_surface)
            alpha = torch.sigmoid(self.alpha_surface) + 0.3
        else:  # coupling
            K = torch.exp(self.log_K_coupling)
            alpha = torch.sigmoid(self.alpha_coupling) + 0.3

        # Flow equation: Q = K * sign(Δh) * |Δh|^alpha
        # This is a generalized form covering:
        # - alpha=0.5: orifice/weir flow (Q ∝ √Δh)
        # - alpha=1.0: Darcy/linear flow (Q ∝ Δh)
        sign = torch.sign(delta_h)
        magnitude = K * torch.pow(torch.abs(delta_h) + 1e-8, alpha)

        # Bound flow to prevent numerical issues
        flow = sign * torch.clamp(magnitude, max=self.max_flow)

        return flow


class SoftBoundaryLoss(nn.Module):
    """Soft boundary constraint for water level predictions using Huber-style penalty.

    Instead of hard constraints (tanh/sigmoid) which cause vanishing gradients,
    this adds a penalty for predictions outside valid range.

    Uses Huber-style penalty to prevent training instability:
    - Quadratic penalty for small violations (|v| <= delta): v²
    - Linear penalty for large violations (|v| > delta): delta * (2|v| - delta)

    Benefits:
    - Normal gradient flow for in-range predictions
    - Smooth quadratic penalty for small violations
    - Linear (bounded gradient) for large violations - prevents loss explosion
    """

    def __init__(self, min_val: float, max_val: float, weight: float = 1.0, delta: float = 10.0):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val
        self.weight = weight
        self.delta = delta  # Huber threshold

    def _huber_penalty(self, violation: torch.Tensor) -> torch.Tensor:
        """Apply Huber-style penalty to violations."""
        abs_v = violation.abs()
        quadratic = 0.5 * violation ** 2
        linear = self.delta * (abs_v - 0.5 * self.delta)
        return torch.where(abs_v <= self.delta, quadratic, linear)

    def forward(self, pred: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions tensor of any shape

        Returns:
            Scalar loss penalizing out-of-range predictions
        """
        # Penalty for values below minimum
        below_min = F.relu(self.min_val - pred)
        # Penalty for values above maximum
        above_max = F.relu(pred - self.max_val)

        # Huber-style penalty (prevents explosion from large violations)
        loss = (self._huber_penalty(below_min) + self._huber_penalty(above_max)).mean()

        return self.weight * loss


class SimplifiedPhysicsLoss(nn.Module):
    """DEPRECATED: Original simplified physics loss - caused variance collapse.
    Use LightPhysicsRegularizer instead.
    """

    def __init__(self, dt_seconds: float = 300.0):
        super().__init__()
        self.dt = dt_seconds

    def forward(self, pred_1d, pred_2d, rainfall=None, node_areas_1d=None,
                node_areas_2d=None, edge_index_2d=None):
        device = pred_1d.device
        return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)


class LightPhysicsRegularizer(nn.Module):
    """Ultra-light physics regularizer that preserves variance.

    Key insight from failed experiments: Strong physics constraints caused
    variance collapse (std: 3.24 vs 13.80), leading to 0.5 vs 0.18 score.

    This regularizer uses VERY soft constraints:
    1. Temporal smoothness: Penalize only EXTREME jumps (not all changes)
    2. Peak consistency: Don't over-smooth peaks
    3. No volume balance: This was the main cause of variance collapse

    Design principles:
    - Weights should be 0.0001-0.001 (100x smaller than before)
    - Only penalize outliers, not normal variation
    - Preserve the model's ability to predict peaks and valleys
    """

    def __init__(self):
        super().__init__()

    def forward(self,
                pred_1d: torch.Tensor,  # [batch, horizon, num_1d]
                pred_2d: torch.Tensor,  # [batch, horizon, num_2d]
                target_1d: Optional[torch.Tensor] = None,  # For variance matching
                target_2d: Optional[torch.Tensor] = None,
                edge_index_2d: Optional[torch.Tensor] = None,
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            temporal_reg: Soft temporal regularization
            variance_reg: Variance preservation regularization
        """
        device = pred_1d.device
        batch, horizon, num_1d = pred_1d.shape
        num_2d = pred_2d.shape[2]

        # === Temporal Smoothness (very soft) ===
        # Only penalize EXTREME jumps using Huber loss with large delta
        # This allows normal variation while preventing unrealistic spikes
        temporal_diff_1d = pred_1d[:, 1:] - pred_1d[:, :-1]  # [batch, horizon-1, num_1d]
        temporal_diff_2d = pred_2d[:, 1:] - pred_2d[:, :-1]

        # Use Huber loss with large delta - only penalizes extreme outliers
        # delta=1.0 means: below delta -> L2/2, above delta -> L1
        # This is much softer than pure L2
        temporal_reg_1d = F.huber_loss(temporal_diff_1d, torch.zeros_like(temporal_diff_1d),
                                        delta=1.0, reduction='mean')
        temporal_reg_2d = F.huber_loss(temporal_diff_2d, torch.zeros_like(temporal_diff_2d),
                                        delta=1.0, reduction='mean')
        temporal_reg = temporal_reg_1d + temporal_reg_2d

        # === Variance Preservation ===
        # If we have targets, encourage predictions to have similar variance
        # This directly prevents variance collapse
        if target_1d is not None and target_2d is not None:
            # Match prediction variance to target variance
            pred_var_1d = pred_1d.var(dim=1).mean()  # Variance across time
            pred_var_2d = pred_2d.var(dim=1).mean()
            target_var_1d = target_1d.var(dim=1).mean()
            target_var_2d = target_2d.var(dim=1).mean()

            # Penalize if prediction variance is too low (variance collapse)
            # Use log ratio for scale-invariance
            var_ratio_1d = torch.log(pred_var_1d + 1e-6) - torch.log(target_var_1d + 1e-6)
            var_ratio_2d = torch.log(pred_var_2d + 1e-6) - torch.log(target_var_2d + 1e-6)

            # Only penalize if prediction variance is LOWER (collapse)
            # Don't penalize higher variance
            variance_reg = F.relu(-var_ratio_1d) + F.relu(-var_ratio_2d)
        else:
            variance_reg = torch.tensor(0.0, device=device)

        return temporal_reg, variance_reg


class SpatialSmoothness(nn.Module):
    """Optional spatial smoothness for 2D surface nodes.

    Uses a soft Laplacian-style regularization that only penalizes
    extreme local gradients, not normal spatial variation.
    """

    def __init__(self, percentile: float = 95.0):
        super().__init__()
        self.percentile = percentile  # Only penalize top 5% of gradients

    def forward(self, pred_2d: torch.Tensor, edge_index_2d: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred_2d: [batch, horizon, num_2d]
            edge_index_2d: [2, num_edges]
        Returns:
            spatial_reg: Soft spatial regularization
        """
        if edge_index_2d is None:
            return torch.tensor(0.0, device=pred_2d.device)

        batch, horizon, num_2d = pred_2d.shape
        src_idx, dst_idx = edge_index_2d[0], edge_index_2d[1]

        # Sample a few timesteps to save memory
        t_samples = list(range(0, horizon, max(1, horizon // 3)))

        spatial_diffs = []
        for t in t_samples:
            h_src = pred_2d[:, t, src_idx]  # [batch, num_edges]
            h_dst = pred_2d[:, t, dst_idx]
            diff = (h_src - h_dst).abs()
            spatial_diffs.append(diff)

        all_diffs = torch.cat(spatial_diffs, dim=1)  # [batch, total_samples]

        # Only penalize extreme gradients (top percentile)
        threshold = torch.quantile(all_diffs, self.percentile / 100.0, dim=1, keepdim=True)
        extreme_diffs = F.relu(all_diffs - threshold)

        return extreme_diffs.mean()


class LocalMassConservationLoss(nn.Module):
    """Computes local (per-node) mass conservation residual with scale normalization.

    For each node i:
        r_i = dV_i/dt - (sum(Q_in) - sum(Q_out)) - S_i

    Where:
        - V_i = A_i * h_i (volume = area * water level)
        - Q_in = sum of flows into node
        - Q_out = sum of flows out of node
        - S_i = source term (rainfall for 2D, inlet flow for 1D)

    Normalizes by reference scale to handle different graph sizes.
    """

    def __init__(self, dt_seconds: float = 300.0, eps: float = 1e-6):
        super().__init__()
        self.dt = dt_seconds  # Timestep in seconds (default 5 minutes)
        self.eps = eps

    def forward(self,
                h_curr: torch.Tensor,  # [batch, nodes]
                h_prev: torch.Tensor,  # [batch, nodes]
                node_areas: torch.Tensor,  # [nodes]
                flows_in: torch.Tensor,  # [batch, nodes] sum of incoming flows
                flows_out: torch.Tensor,  # [batch, nodes] sum of outgoing flows
                sources: torch.Tensor,  # [batch, nodes] source terms
                ) -> torch.Tensor:
        """
        Returns:
            loss: Normalized mean absolute residual across all nodes and batch
        """
        # Normalize node areas to prevent scale issues across different graphs
        mean_area = node_areas.mean().clamp(min=self.eps)
        norm_areas = node_areas / mean_area

        # Volume change: dV/dt (normalized by mean area)
        V_curr = h_curr * norm_areas.unsqueeze(0)  # [batch, nodes]
        V_prev = h_prev * norm_areas.unsqueeze(0)
        dV_dt = (V_curr - V_prev) / self.dt

        # Normalize flows by number of nodes to handle graphs of different sizes
        num_nodes = h_curr.shape[1]
        scale_factor = math.sqrt(num_nodes)  # sqrt scaling for stability

        # Mass balance residual (clamp flows to prevent explosion)
        flows_in_clamped = flows_in.clamp(-1e6, 1e6) / scale_factor
        flows_out_clamped = flows_out.clamp(-1e6, 1e6) / scale_factor
        net_inflow = flows_in_clamped - flows_out_clamped

        # Normalize sources similarly
        sources_norm = sources / scale_factor

        residual = dV_dt - net_inflow - sources_norm

        # Use smooth L1 loss for better gradient behavior
        loss = F.smooth_l1_loss(residual, torch.zeros_like(residual), reduction='mean')

        return loss


class GlobalMassConservationLoss(nn.Module):
    """Computes global (system-wide) mass conservation residual with scale normalization.

    For the entire system:
        R = sum(dV_i/dt) - In_global + Out_global - S_global

    This should be approximately zero for a closed system.
    """

    def __init__(self, dt_seconds: float = 300.0, eps: float = 1e-6):
        super().__init__()
        self.dt = dt_seconds
        self.eps = eps

    def forward(self,
                h_curr: torch.Tensor,  # [batch, nodes]
                h_prev: torch.Tensor,  # [batch, nodes]
                node_areas: torch.Tensor,  # [nodes]
                boundary_inflow: torch.Tensor,  # [batch] total inflow at boundaries
                boundary_outflow: torch.Tensor,  # [batch] total outflow at boundaries
                total_source: torch.Tensor,  # [batch] total source (rainfall, inlet)
                ) -> torch.Tensor:
        """
        Returns:
            loss: Normalized absolute global residual
        """
        num_nodes = h_curr.shape[1]

        # Normalize areas
        mean_area = node_areas.mean().clamp(min=self.eps)
        norm_areas = node_areas / mean_area

        # Total volume change (normalized)
        V_curr = (h_curr * norm_areas.unsqueeze(0)).sum(dim=1)  # [batch]
        V_prev = (h_prev * norm_areas.unsqueeze(0)).sum(dim=1)
        dV_dt_total = (V_curr - V_prev) / self.dt

        # Normalize by number of nodes for cross-graph comparability
        scale_factor = math.sqrt(num_nodes)
        dV_dt_norm = dV_dt_total / scale_factor
        boundary_inflow_norm = boundary_inflow.clamp(-1e6, 1e6) / scale_factor
        boundary_outflow_norm = boundary_outflow.clamp(-1e6, 1e6) / scale_factor
        total_source_norm = total_source / scale_factor

        # Global residual
        residual = dV_dt_norm - boundary_inflow_norm + boundary_outflow_norm - total_source_norm

        # Use smooth L1 loss
        loss = F.smooth_l1_loss(residual, torch.zeros_like(residual), reduction='mean')

        return loss


class PhysicsResidualLoss(nn.Module):
    """Scale-normalized conservation residual in physical space.

    For each node i:
        r_i = A_i * (h_i(t) - h_i(t-1)) / dt - (Q_in,i - Q_out,i) - S_i

    The loss is computed on a normalized residual so it remains stable across
    model scales (Model 1 vs Model 2) and rollout phases.
    """

    def __init__(self, dt_seconds: float = 300.0, huber_delta: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.dt = dt_seconds
        self.huber_delta = huber_delta
        self.eps = eps

    def forward(
        self,
        h_curr: torch.Tensor,      # [batch, nodes]
        h_prev: torch.Tensor,      # [batch, nodes]
        node_areas: torch.Tensor,  # [nodes]
        flows_in: torch.Tensor,    # [batch, nodes]
        flows_out: torch.Tensor,   # [batch, nodes]
        sources: torch.Tensor,     # [batch, nodes]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns local residual loss, global residual loss, and raw global residual."""
        area = node_areas.unsqueeze(0).clamp(min=self.eps)
        dV_dt = (h_curr - h_prev) * area / self.dt
        rhs = flows_in - flows_out + sources

        # Local normalized residual.
        scale_local = (dV_dt.abs().mean(dim=1, keepdim=True) + rhs.abs().mean(dim=1, keepdim=True)).clamp(min=self.eps)
        residual_local = (dV_dt - rhs) / scale_local
        local_loss = F.huber_loss(
            residual_local,
            torch.zeros_like(residual_local),
            delta=self.huber_delta,
            reduction='mean',
        )

        # Global normalized residual.
        residual_global = dV_dt.sum(dim=1) - rhs.sum(dim=1)
        scale_global = (dV_dt.sum(dim=1).abs() + rhs.sum(dim=1).abs()).clamp(min=self.eps)
        residual_global_norm = residual_global / scale_global
        global_loss = F.huber_loss(
            residual_global_norm,
            torch.zeros_like(residual_global_norm),
            delta=self.huber_delta,
            reduction='mean',
        )

        global_abs = residual_global.abs().mean()
        return local_loss, global_loss, global_abs


class CurriculumScheduler:
    """Scheduler for curriculum learning with progressive rollout lengths.

    Starts with short rollouts (1 step) and progressively increases to full horizon.
    Only advances to next stage when validation converges.
    """

    def __init__(self,
                 stages: List[int] = [1, 4, 8, 16, 32, 90],
                 epochs_per_stage: int = 5,
                 patience_per_stage: int = 3,
                 reset_best_on_advance: bool = True):
        """
        Args:
            stages: List of rollout lengths, increasing order
            epochs_per_stage: Minimum epochs per stage before advancing
            patience_per_stage: Epochs without improvement to wait before advancing
            reset_best_on_advance: Reset stage best metric when advancing stage
        """
        self.stages = stages
        self.epochs_per_stage = epochs_per_stage
        self.patience_per_stage = patience_per_stage
        self.reset_best_on_advance = reset_best_on_advance

        self.current_stage_idx = 0
        self.epochs_in_stage = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0

    @property
    def current_rollout_len(self) -> int:
        return self.stages[self.current_stage_idx]

    @property
    def is_final_stage(self) -> bool:
        return self.current_stage_idx >= len(self.stages) - 1

    def step(self, val_loss: float) -> bool:
        """Update scheduler based on validation loss.

        Returns:
            advanced: True if moved to next stage
        """
        self.epochs_in_stage += 1
        advanced = False

        # Track best loss in current stage
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1

        # Check if we should advance to next stage
        if not self.is_final_stage:
            min_epochs_met = self.epochs_in_stage >= self.epochs_per_stage
            patience_exhausted = self.patience_counter >= self.patience_per_stage

            if min_epochs_met and patience_exhausted:
                self.current_stage_idx += 1
                self.epochs_in_stage = 0
                self.patience_counter = 0
                # Stage metrics are not directly comparable across rollout lengths.
                if self.reset_best_on_advance:
                    self.best_val_loss = float('inf')
                advanced = True
                print(f"\n>>> Curriculum: Advanced to stage {self.current_stage_idx + 1}/{len(self.stages)}, "
                      f"rollout_len={self.current_rollout_len}")

        return advanced

    def state_dict(self) -> Dict:
        return {
            'current_stage_idx': self.current_stage_idx,
            'epochs_in_stage': self.epochs_in_stage,
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter,
            'reset_best_on_advance': self.reset_best_on_advance,
        }

    def load_state_dict(self, state: Dict):
        self.current_stage_idx = state['current_stage_idx']
        self.epochs_in_stage = state['epochs_in_stage']
        self.best_val_loss = state['best_val_loss']
        self.patience_counter = state['patience_counter']
        self.reset_best_on_advance = state.get('reset_best_on_advance', True)


# =============================================================================
# VGSSM COMPONENTS
# =============================================================================

class HeteroGNNBlock(nn.Module):
    """Lightweight heterogeneous GNN block for latent state transitions."""

    def __init__(self, latent_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim

        self.edge_types = [
            ('1d', 'pipe', '1d'),
            ('2d', 'surface', '2d'),
            ('1d', 'couples_to', '2d'),
            ('2d', 'couples_from', '1d'),
        ]

        conv_dict = {}
        for edge_type in self.edge_types:
            conv_dict[edge_type] = SAGEConv((latent_dim, latent_dim), hidden_dim, aggr='mean')
        self.conv = HeteroConv(conv_dict, aggr='sum')

        self.proj_1d = nn.Linear(hidden_dim, latent_dim)
        self.proj_2d = nn.Linear(hidden_dim, latent_dim)
        self.norm_1d = nn.LayerNorm(latent_dim)
        self.norm_2d = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(dropout)
        # Cache expanded block-diagonal edge indices keyed by graph identity and batch shape.
        self._batched_edge_index_cache: Dict[Tuple[Any, int, int, int, str], torch.Tensor] = {}

    def _expand_edge_index_for_batch(
        self,
        edge_index: torch.Tensor,
        batch_size: int,
        num_src: int,
        num_dst: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Create block-diagonal edge index for batched disjoint graphs."""
        device_key = str(device)
        cache_key = (edge_index.data_ptr(), batch_size, num_src, num_dst, device_key)
        cached = self._batched_edge_index_cache.get(cache_key)
        if cached is not None:
            return cached

        num_edges = edge_index.shape[1]
        src = edge_index[0].to(device=device)
        dst = edge_index[1].to(device=device)

        batch_offsets_src = (
            torch.arange(batch_size, device=device, dtype=src.dtype)
            .repeat_interleave(num_edges)
            * num_src
        )
        batch_offsets_dst = (
            torch.arange(batch_size, device=device, dtype=dst.dtype)
            .repeat_interleave(num_edges)
            * num_dst
        )

        batched_src = src.repeat(batch_size) + batch_offsets_src
        batched_dst = dst.repeat(batch_size) + batch_offsets_dst
        expanded = torch.stack([batched_src, batched_dst], dim=0)
        self._batched_edge_index_cache[cache_key] = expanded
        return expanded

    def forward(self, z_1d, z_2d, edge_index_dict):
        batch_size = z_1d.shape[0]
        num_1d = z_1d.shape[1]
        num_2d = z_2d.shape[1]
        device = z_1d.device

        x_dict = {
            '1d': z_1d.reshape(batch_size * num_1d, -1),
            '2d': z_2d.reshape(batch_size * num_2d, -1),
        }

        batched_edge_index_dict = {}
        for edge_type in self.edge_types:
            if edge_type not in edge_index_dict:
                continue
            src_type, _, dst_type = edge_type
            src_nodes = num_1d if src_type == '1d' else num_2d
            dst_nodes = num_1d if dst_type == '1d' else num_2d
            batched_edge_index_dict[edge_type] = self._expand_edge_index_for_batch(
                edge_index_dict[edge_type], batch_size, src_nodes, dst_nodes, device
            )

        out_dict = self.conv(x_dict, batched_edge_index_dict)
        hidden_dim_1d = self.proj_1d.in_features
        hidden_dim_2d = self.proj_2d.in_features
        out_1d = out_dict.get(
            '1d',
            torch.zeros(batch_size * num_1d, hidden_dim_1d, device=device, dtype=z_1d.dtype),
        ).view(batch_size, num_1d, hidden_dim_1d)
        out_2d = out_dict.get(
            '2d',
            torch.zeros(batch_size * num_2d, hidden_dim_2d, device=device, dtype=z_2d.dtype),
        ).view(batch_size, num_2d, hidden_dim_2d)

        delta_1d = self.proj_1d(out_1d)
        delta_2d = self.proj_2d(out_2d)
        delta_1d = self.dropout(self.norm_1d(delta_1d))
        delta_2d = self.dropout(self.norm_2d(delta_2d))

        return delta_1d, delta_2d


class LatentTransition(nn.Module):
    """Latent state transition: z_{t+1} = z_t + f(z_t, graph, u_t, c_e)"""

    def __init__(self, latent_dim, hidden_dim, c_e_dim, control_dim_1d, control_dim_2d,
                 num_gnn_layers=2, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim

        self.gnn_blocks = nn.ModuleList([
            HeteroGNNBlock(latent_dim, hidden_dim, dropout)
            for _ in range(num_gnn_layers)
        ])

        self.temporal_mlp_1d = nn.Sequential(
            nn.Linear(latent_dim + c_e_dim + control_dim_1d, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.temporal_mlp_2d = nn.Sequential(
            nn.Linear(latent_dim + c_e_dim + control_dim_2d, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.gate_1d = nn.Sequential(nn.Linear(latent_dim * 2, latent_dim), nn.Sigmoid())
        self.gate_2d = nn.Sequential(nn.Linear(latent_dim * 2, latent_dim), nn.Sigmoid())

    def forward(self, z_1d, z_2d, edge_index_dict, u_1d, u_2d, c_e):
        batch, num_1d, _ = z_1d.shape
        num_2d = z_2d.shape[1]

        delta_spatial_1d = torch.zeros_like(z_1d)
        delta_spatial_2d = torch.zeros_like(z_2d)

        z_1d_curr = z_1d
        z_2d_curr = z_2d
        for gnn_block in self.gnn_blocks:
            d1d, d2d = gnn_block(z_1d_curr, z_2d_curr, edge_index_dict)
            delta_spatial_1d = delta_spatial_1d + d1d
            delta_spatial_2d = delta_spatial_2d + d2d
            z_1d_curr = z_1d_curr + d1d
            z_2d_curr = z_2d_curr + d2d

        c_e_1d = c_e.unsqueeze(1).expand(-1, num_1d, -1)
        c_e_2d = c_e.unsqueeze(1).expand(-1, num_2d, -1)

        if u_1d is not None:
            input_1d = torch.cat([z_1d, c_e_1d, u_1d], dim=-1)
        else:
            input_1d = torch.cat([z_1d, c_e_1d], dim=-1)
            zeros = torch.zeros(batch, num_1d, self.temporal_mlp_1d[0].in_features - input_1d.shape[-1], device=z_1d.device)
            input_1d = torch.cat([input_1d, zeros], dim=-1)

        delta_temporal_1d = self.temporal_mlp_1d(input_1d)

        input_2d = torch.cat([z_2d, c_e_2d, u_2d], dim=-1)
        delta_temporal_2d = self.temporal_mlp_2d(input_2d)

        delta_1d = delta_spatial_1d + delta_temporal_1d
        delta_2d = delta_spatial_2d + delta_temporal_2d

        gate_1d = self.gate_1d(torch.cat([z_1d, delta_1d], dim=-1))
        gate_2d = self.gate_2d(torch.cat([z_2d, delta_2d], dim=-1))

        z_1d_next = z_1d + gate_1d * delta_1d
        z_2d_next = z_2d + gate_2d * delta_2d

        return z_1d_next, z_2d_next


class PhysicsConstrainedTransition(nn.Module):
    """
    Physics-Constrained Transition: z_{t+1} = z_t + delta_neural + alpha * delta_physics

    Scientific Design:
    - Neural transition learns flexible dynamics from data
    - Physics transition enforces mass conservation constraints
    - Learnable blending weight alpha balances flexibility vs physics

    The physics delta is computed by:
    1. Decode z_t -> water levels h_t
    2. Compute physics-based update: h_{t+1} = h_t + dt * (Q_in - Q_out) / A
    3. Encode change: delta_physics = encode(h_{t+1} - h_t)
    """

    def __init__(self, latent_dim, hidden_dim, c_e_dim, control_dim_1d, control_dim_2d,
                 num_gnn_layers=2, dropout=0.1, dt_seconds=300.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.dt = dt_seconds

        # Neural transition (same as LatentTransition)
        self.gnn_blocks = nn.ModuleList([
            HeteroGNNBlock(latent_dim, hidden_dim, dropout)
            for _ in range(num_gnn_layers)
        ])

        self.temporal_mlp_1d = nn.Sequential(
            nn.Linear(latent_dim + c_e_dim + control_dim_1d, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.temporal_mlp_2d = nn.Sequential(
            nn.Linear(latent_dim + c_e_dim + control_dim_2d, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.gate_1d = nn.Sequential(nn.Linear(latent_dim * 2, latent_dim), nn.Sigmoid())
        self.gate_2d = nn.Sequential(nn.Linear(latent_dim * 2, latent_dim), nn.Sigmoid())

        # Physics components
        # Decoder: latent -> water level
        self.h_decoder_1d = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.h_decoder_2d = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # Flow coefficient: determines hydraulic resistance per edge
        self.flow_coef_1d = nn.Parameter(torch.tensor(1.0))
        self.flow_coef_2d = nn.Parameter(torch.tensor(0.5))
        self.flow_coef_coupling = nn.Parameter(torch.tensor(0.3))

        # Encoder: water level change -> latent change
        self.dh_encoder_1d = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.dh_encoder_2d = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Learnable physics weight (starts small, can grow during training)
        self.physics_weight = nn.Parameter(torch.tensor(-2.0))  # sigmoid(-2) ≈ 0.12
        self.max_flow = 5.0

    def _compute_flow(self, h_src, h_dst, flow_coef):
        """Compute flow from head difference using orifice equation: Q = K * sign(Δh) * sqrt(|Δh|)"""
        delta_h = h_src - h_dst
        flow = flow_coef * torch.sign(delta_h) * torch.sqrt(torch.abs(delta_h) + 1e-6)
        return flow.clamp(-self.max_flow, self.max_flow)

    def _compute_physics_delta(self, z_1d, z_2d, edge_index_dict, u_1d, u_2d):
        """Compute physics-based water level change using mass conservation."""
        batch, num_1d, _ = z_1d.shape
        num_2d = z_2d.shape[1]
        device = z_1d.device

        # Decode to water levels
        h_1d = self.h_decoder_1d(z_1d).squeeze(-1)  # [batch, num_1d]
        h_2d = self.h_decoder_2d(z_2d).squeeze(-1)  # [batch, num_2d]

        # Initialize net flow (inflow - outflow)
        net_flow_1d = torch.zeros_like(h_1d)
        net_flow_2d = torch.zeros_like(h_2d)

        # Process each edge type
        for edge_type, edge_index in edge_index_dict.items():
            src_type, edge_name, dst_type = edge_type
            if edge_index.shape[1] == 0:
                continue

            # Get flow coefficient
            if edge_name == 'pipe':
                flow_coef = torch.sigmoid(self.flow_coef_1d)
                h_src = h_1d[:, edge_index[0]]
                h_dst = h_1d[:, edge_index[1]]
            elif edge_name == 'surface':
                flow_coef = torch.sigmoid(self.flow_coef_2d)
                h_src = h_2d[:, edge_index[0]]
                h_dst = h_2d[:, edge_index[1]]
            else:  # coupling
                flow_coef = torch.sigmoid(self.flow_coef_coupling)
                if src_type == '1d':
                    h_src = h_1d[:, edge_index[0]]
                    h_dst = h_2d[:, edge_index[1]]
                else:
                    h_src = h_2d[:, edge_index[0]]
                    h_dst = h_1d[:, edge_index[1]]

            # Compute flow
            flow = self._compute_flow(h_src, h_dst, flow_coef)

            # Accumulate net flow
            if edge_name == 'pipe':
                net_flow_1d.scatter_add_(1, edge_index[1].unsqueeze(0).expand(batch, -1), flow)
                net_flow_1d.scatter_add_(1, edge_index[0].unsqueeze(0).expand(batch, -1), -flow)
            elif edge_name == 'surface':
                net_flow_2d.scatter_add_(1, edge_index[1].unsqueeze(0).expand(batch, -1), flow)
                net_flow_2d.scatter_add_(1, edge_index[0].unsqueeze(0).expand(batch, -1), -flow)
            else:  # coupling
                if src_type == '1d':
                    net_flow_1d.scatter_add_(1, edge_index[0].unsqueeze(0).expand(batch, -1), -flow)
                    net_flow_2d.scatter_add_(1, edge_index[1].unsqueeze(0).expand(batch, -1), flow)
                else:
                    net_flow_2d.scatter_add_(1, edge_index[0].unsqueeze(0).expand(batch, -1), -flow)
                    net_flow_1d.scatter_add_(1, edge_index[1].unsqueeze(0).expand(batch, -1), flow)

        # Add rainfall as source term (rainfall adds to water level)
        net_flow_1d = net_flow_1d + u_1d.squeeze(-1)
        net_flow_2d = net_flow_2d + u_2d.squeeze(-1)

        # Compute water level change: dh = dt * net_flow (simplified, assumes unit area)
        dh_1d = (self.dt / 1000.0) * net_flow_1d  # Scale down for stability
        dh_2d = (self.dt / 1000.0) * net_flow_2d

        # Clamp to prevent explosion
        dh_1d = dh_1d.clamp(-0.5, 0.5)
        dh_2d = dh_2d.clamp(-0.5, 0.5)

        # Encode to latent delta
        delta_physics_1d = self.dh_encoder_1d(dh_1d.unsqueeze(-1))
        delta_physics_2d = self.dh_encoder_2d(dh_2d.unsqueeze(-1))

        return delta_physics_1d, delta_physics_2d

    def forward(self, z_1d, z_2d, edge_index_dict, u_1d, u_2d, c_e):
        batch, num_1d, _ = z_1d.shape
        num_2d = z_2d.shape[1]
        device = z_1d.device

        # Handle None inputs
        if u_1d is None:
            u_1d = torch.zeros(batch, num_1d, 1, device=device)
        if u_2d is None:
            u_2d = torch.zeros(batch, num_2d, 1, device=device)

        # --- Neural transition (same as LatentTransition) ---
        delta_spatial_1d = torch.zeros_like(z_1d)
        delta_spatial_2d = torch.zeros_like(z_2d)

        for gnn_block in self.gnn_blocks:
            d1d, d2d = gnn_block(z_1d, z_2d, edge_index_dict)
            delta_spatial_1d = delta_spatial_1d + d1d
            delta_spatial_2d = delta_spatial_2d + d2d

        c_e_1d = c_e.unsqueeze(1).expand(-1, num_1d, -1)
        c_e_2d = c_e.unsqueeze(1).expand(-1, num_2d, -1)

        input_1d = torch.cat([z_1d, c_e_1d, u_1d], dim=-1)
        delta_temporal_1d = self.temporal_mlp_1d(input_1d)

        input_2d = torch.cat([z_2d, c_e_2d, u_2d], dim=-1)
        delta_temporal_2d = self.temporal_mlp_2d(input_2d)

        delta_neural_1d = delta_spatial_1d + delta_temporal_1d
        delta_neural_2d = delta_spatial_2d + delta_temporal_2d

        gate_1d = self.gate_1d(torch.cat([z_1d, delta_neural_1d], dim=-1))
        gate_2d = self.gate_2d(torch.cat([z_2d, delta_neural_2d], dim=-1))

        delta_neural_1d = gate_1d * delta_neural_1d
        delta_neural_2d = gate_2d * delta_neural_2d

        # --- Physics transition ---
        delta_physics_1d, delta_physics_2d = self._compute_physics_delta(
            z_1d, z_2d, edge_index_dict, u_1d, u_2d
        )

        # Blend neural + physics
        alpha = torch.sigmoid(self.physics_weight)  # Learnable weight in [0, 1]
        z_1d_next = z_1d + delta_neural_1d + alpha * delta_physics_1d
        z_2d_next = z_2d + delta_neural_2d + alpha * delta_physics_2d

        return z_1d_next, z_2d_next


class LatentInferenceNet(nn.Module):
    """Inference network for initial latent state: q(z_0 | prefix, c_e)

    Original GRU-based version (fallback when Timer is disabled).
    """

    def __init__(self, input_dim, hidden_dim, latent_dim, c_e_dim, spatial_dim, num_nodes, dropout=0.1):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_nodes = num_nodes

        self.input_proj = nn.Linear(input_dim + spatial_dim + c_e_dim, hidden_dim)
        self.gru = nn.GRU(
            input_size=hidden_dim, hidden_size=hidden_dim,
            num_layers=2, batch_first=True, dropout=dropout,
        )
        self.mu_proj = nn.Linear(hidden_dim, latent_dim)
        self.logvar_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(self, prefix_seq, spatial_emb, c_e):
        batch, prefix_len, num_nodes, input_dim = prefix_seq.shape

        spatial_expanded = spatial_emb.unsqueeze(0).unsqueeze(0).expand(batch, prefix_len, -1, -1)
        c_e_expanded = c_e.unsqueeze(1).unsqueeze(2).expand(-1, prefix_len, num_nodes, -1)

        combined = torch.cat([prefix_seq, spatial_expanded, c_e_expanded], dim=-1)
        h = self.input_proj(combined)
        h = h.permute(0, 2, 1, 3).reshape(batch * num_nodes, prefix_len, -1)

        _, h_final = self.gru(h)
        h_final = h_final[-1]

        mu = self.mu_proj(h_final)
        logvar = self.logvar_proj(h_final)
        logvar = torch.clamp(logvar, min=-10, max=2)

        mu = mu.view(batch, num_nodes, self.latent_dim)
        logvar = logvar.view(batch, num_nodes, self.latent_dim)

        return mu, logvar


class TimerPosterior(nn.Module):
    """
    Timer 3.0 style posterior estimation for z_0 using causal Transformer.

    This replaces the GRU-based LatentInferenceNet with a GPT-style
    causal transformer for better temporal representation learning.

    Key benefits over GRU:
    - Better long-range dependency modeling
    - Parallelizable during training
    - More expressive temporal representations

    Architecture:
        prefix_seq -> Input Projection -> Positional Encoding
        -> N x CausalTransformerBlock -> LayerNorm -> Pooling -> (μ, σ)
    """

    def __init__(self, input_dim, hidden_dim, latent_dim, c_e_dim, spatial_dim,
                 num_nodes, num_layers=4, num_heads=4, dropout=0.1, max_seq_len=64):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes

        # Input projection: combine prefix features + spatial + c_e
        self.input_proj = nn.Linear(input_dim + spatial_dim + c_e_dim, hidden_dim)

        # Learnable positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Causal transformer blocks
        self.blocks = nn.ModuleList([
            TimerBlock(hidden_dim, num_heads, ffn_ratio=4, dropout=dropout, max_seq_len=max_seq_len)
            for _ in range(num_layers)
        ])

        self.ln_out = nn.LayerNorm(hidden_dim)

        # Output heads for μ and log(σ²)
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.logvar_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, prefix_seq, spatial_emb, c_e):
        """
        Args:
            prefix_seq: [batch, prefix_len, num_nodes, input_dim] - observed sequence
            spatial_emb: [num_nodes, spatial_dim] - spatial node embeddings
            c_e: [batch, c_e_dim] - event latent

        Returns:
            mu: [batch, num_nodes, latent_dim] - posterior mean
            logvar: [batch, num_nodes, latent_dim] - posterior log variance
        """
        batch, prefix_len, num_nodes, input_dim = prefix_seq.shape

        # Expand spatial and c_e to match prefix sequence dimensions
        spatial_expanded = spatial_emb.unsqueeze(0).unsqueeze(0).expand(batch, prefix_len, -1, -1)
        c_e_expanded = c_e.unsqueeze(1).unsqueeze(2).expand(-1, prefix_len, num_nodes, -1)

        # Concatenate features
        combined = torch.cat([prefix_seq, spatial_expanded, c_e_expanded], dim=-1)

        # Project to hidden dimension
        h = self.input_proj(combined)  # [batch, prefix_len, num_nodes, hidden_dim]

        # Reshape to process each node's sequence through transformer
        # [batch, prefix_len, num_nodes, hidden_dim] -> [batch * num_nodes, prefix_len, hidden_dim]
        h = h.permute(0, 2, 1, 3).reshape(batch * num_nodes, prefix_len, self.hidden_dim)

        # Add positional encoding
        h = h + self.pos_embed[:, :prefix_len, :]

        # Pass through causal transformer blocks
        for block in self.blocks:
            h = block(h)

        h = self.ln_out(h)

        # Use last position output (contains full context due to causal attention)
        h_final = h[:, -1, :]  # [batch * num_nodes, hidden_dim]

        # Predict posterior parameters
        mu = self.mu_head(h_final)  # [batch * num_nodes, latent_dim]
        logvar = self.logvar_head(h_final)  # [batch * num_nodes, latent_dim]

        # Clamp logvar for numerical stability
        logvar = torch.clamp(logvar, min=-10, max=2)

        # Reshape to [batch, num_nodes, latent_dim]
        mu = mu.view(batch, num_nodes, self.latent_dim)
        logvar = logvar.view(batch, num_nodes, self.latent_dim)

        return mu, logvar


class LatentDecoder(nn.Module):
    """Decodes latent state to water level predictions.

    Supports three modes:
    - Absolute: Directly predicts h_t
    - Delta: Predicts delta_h, final h_t = h_{t-1} + delta_h
    - Baseline residual: Predicts residual, final h_t = baseline + residual

    The baseline residual mode is the most stable for bounded outputs:
    - Model predicts small deviations from per-node mean
    - No explicit penalty needed - residuals naturally stay small
    - No vanishing gradients like tanh/sigmoid bounds
    """

    def __init__(self, latent_dim, spatial_dim, hidden_dim, output_dim,
                 delta_mode: bool = False, dropout: float = 0.1,
                 output_min: float = None, output_max: float = None,
                 use_baseline_residual: bool = False, num_nodes: int = None,
                 use_sigmoid_bounds: bool = False):
        super().__init__()
        self.delta_mode = delta_mode
        self.output_dim = output_dim
        self.use_baseline_residual = use_baseline_residual
        self.use_sigmoid_bounds = use_sigmoid_bounds

        # Store bounds for reference
        self.output_min = output_min
        self.output_max = output_max

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + spatial_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        if delta_mode:
            # Scale for delta prediction (small changes expected)
            self.delta_scale = nn.Parameter(torch.ones(1) * 0.1)

        if use_baseline_residual and num_nodes is not None:
            # Per-node baseline (initialized to middle of range, updated from data)
            if output_min is not None and output_max is not None:
                init_baseline = (output_min + output_max) / 2
                # Max deviation covers full data range from baseline
                # Using 1.5x half-range to allow some extrapolation
                self.max_deviation = (output_max - output_min) / 2 * 1.5
            else:
                init_baseline = 0.0
                self.max_deviation = 10.0  # Default if no bounds specified
            # Use buffer so it moves with the model but isn't trained
            self.register_buffer('node_baseline', torch.full((num_nodes, 1), init_baseline))

    def set_baseline_from_data(self, mean_per_node: torch.Tensor):
        """Set per-node baseline from training data statistics."""
        if self.use_baseline_residual and hasattr(self, 'node_baseline'):
            self.node_baseline.copy_(mean_per_node.view(-1, 1))

    def forward(self, z: torch.Tensor, spatial_emb: torch.Tensor,
                h_prev: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: Latent state [batch, nodes, latent_dim]
            spatial_emb: Spatial embeddings [nodes, spatial_dim]
            h_prev: Previous water level [batch, nodes, 1] (required if delta_mode)

        Returns:
            h: Water level prediction [batch, nodes, output_dim]
            delta_h: Change in water level (zeros if not delta_mode) [batch, nodes, output_dim]
        """
        batch = z.shape[0]
        spatial_expanded = spatial_emb.unsqueeze(0).expand(batch, -1, -1)
        combined = torch.cat([z, spatial_expanded], dim=-1)

        raw_output = self.decoder(combined)

        if self.delta_mode and h_prev is not None:
            delta_h = raw_output * self.delta_scale
            h = h_prev + delta_h
        elif self.use_sigmoid_bounds and self.output_min is not None and self.output_max is not None:
            # Sigmoid scaling: maps unbounded output to [min, max]
            # sigmoid(x) -> [0, 1], then scale to [min, max]
            # This is the cleanest bounded output approach:
            # - No penalty terms that compete with reconstruction loss
            # - Smooth gradients everywhere (no saturation issues like tanh at extremes)
            # - Output is ALWAYS within bounds by construction
            h = self.output_min + torch.sigmoid(raw_output) * (self.output_max - self.output_min)
            delta_h = torch.zeros_like(raw_output)
        elif self.use_baseline_residual and hasattr(self, 'node_baseline'):
            # Bounded residual: baseline + tanh(raw) * max_deviation
            # tanh bounds output to [-1, 1], max_deviation scales to actual data range
            # This ensures predictions stay bounded while allowing full data variation
            residual = torch.tanh(raw_output) * self.max_deviation
            h = self.node_baseline.unsqueeze(0) + residual  # [1, nodes, 1] + [batch, nodes, 1]
            delta_h = residual  # Track the residual as delta
        else:
            # Absolute prediction
            h = raw_output
            delta_h = torch.zeros_like(raw_output)

        return h, delta_h


# ==============================================================================
# Physics-Based Decoder (Mass-Conserving)
# ==============================================================================

class PhysicsDecoder(nn.Module):
    """Physics-based decoder that derives water levels from mass conservation.

    Instead of directly predicting water levels h, this decoder:
    1. Predicts flow coefficients K from latent states
    2. Computes flows Q using hydraulic equations: Q = K * sign(Δh) * √|Δh|
    3. Updates volumes via mass balance: dV = (ΣQ_in - ΣQ_out + R) * dt
    4. Derives water levels from volumes: h = V / A

    This ensures:
    - Mass conservation is satisfied by construction
    - Water levels are naturally bounded by storage geometry
    - No penalty terms needed for physical constraints
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        num_nodes: int,
        node_areas: torch.Tensor = None,  # Storage area per node [m²]
        min_depth: float = 0.0,           # Minimum water depth [m]
        max_depth: float = 10.0,          # Maximum water depth [m]
        base_elevation: float = 0.0,      # Base elevation for water level
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.base_elevation = base_elevation

        # Default node areas (1 m² per node if not specified)
        if node_areas is not None:
            self.register_buffer('node_area', node_areas)
        else:
            self.register_buffer('node_area', torch.ones(num_nodes))

        # Flow coefficient predictor: z_src, z_dst -> K (always positive)
        # K represents hydraulic conductivity/roughness for each edge
        self.flow_coeff_net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softplus(),  # Ensures K > 0
        )

        # Learnable base flow coefficient (scaling factor)
        self.base_K = nn.Parameter(torch.ones(1) * 0.1)

        # Volume initialization network: latent -> initial volume estimate
        self.volume_init_net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),  # Volume > 0
        )

    def compute_flows(
        self,
        z: torch.Tensor,           # [batch, num_nodes, latent_dim]
        h: torch.Tensor,           # [batch, num_nodes, 1] - current water levels
        edge_index: torch.Tensor,  # [2, num_edges]
    ) -> torch.Tensor:
        """Compute flows on edges using hydraulic equation.

        Q = K * sign(Δh) * √|Δh|  (orifice/weir flow approximation)

        Memory-efficient version: uses fixed flow coefficient (not per-edge learned)
        This is more physically correct anyway - K depends on pipe/surface properties
        which are fixed, not on the latent state.

        Returns:
            flows: [batch, num_edges] - flow rate on each edge (positive = src->dst)
        """
        src, dst = edge_index[0], edge_index[1]

        # Compute head difference (memory efficient - no large edge tensors)
        h_src = h[:, src, 0]  # [batch, num_edges]
        h_dst = h[:, dst, 0]  # [batch, num_edges]
        delta_h = h_src - h_dst

        # Use fixed flow coefficient (learnable scalar, not per-edge)
        # This is physically correct: K depends on geometry, not state
        K = self.base_K  # Scalar

        # Hydraulic flow equation: Q = K * sign(Δh) * √|Δh|
        # This approximates orifice/weir flow behavior
        eps = 1e-6
        Q = K * torch.sign(delta_h) * torch.sqrt(torch.abs(delta_h) + eps)

        return Q  # [batch, num_edges]

    def mass_balance_update(
        self,
        V: torch.Tensor,           # [batch, num_nodes, 1] - current volumes
        Q: torch.Tensor,           # [batch, num_edges] - flows
        edge_index: torch.Tensor,  # [2, num_edges]
        source: torch.Tensor,      # [batch, num_nodes, 1] - source terms (rainfall)
        dt: float = 300.0,         # Time step [seconds]
    ) -> torch.Tensor:
        """Update volumes using mass balance.

        dV_i/dt = Σ Q_in - Σ Q_out + S_i
        V_new = V + dV * dt

        Returns:
            V_new: [batch, num_nodes, 1] - updated volumes
        """
        batch, num_nodes, _ = V.shape
        src, dst = edge_index[0], edge_index[1]
        device = V.device

        # Compute net flow at each node
        # Q > 0 means flow from src to dst
        # So dst receives Q (inflow), src loses Q (outflow)
        Q_in = torch.zeros(batch, num_nodes, device=device)
        Q_out = torch.zeros(batch, num_nodes, device=device)

        # Aggregate flows
        Q_in.scatter_add_(1, dst.unsqueeze(0).expand(batch, -1), Q)
        Q_out.scatter_add_(1, src.unsqueeze(0).expand(batch, -1), Q)

        # Net flow (positive = net inflow)
        Q_net = Q_in - Q_out  # [batch, num_nodes]

        # Mass balance: dV/dt = Q_net + source
        dV = (Q_net + source.squeeze(-1)) * dt  # [batch, num_nodes]

        # Update volume (ensure non-negative)
        V_new = (V.squeeze(-1) + dV).clamp(min=0)  # [batch, num_nodes]

        return V_new.unsqueeze(-1)  # [batch, num_nodes, 1]

    def volume_to_depth(self, V: torch.Tensor) -> torch.Tensor:
        """Convert volume to water depth.

        depth = V / A, bounded by [min_depth, max_depth]

        Returns:
            depth: [batch, num_nodes, 1]
        """
        # depth = volume / area
        depth = V / (self.node_area.unsqueeze(0).unsqueeze(-1) + 1e-6)

        # Bound by physical limits
        depth = depth.clamp(min=self.min_depth, max=self.max_depth)

        return depth

    def depth_to_level(self, depth: torch.Tensor) -> torch.Tensor:
        """Convert water depth to water level (add base elevation).

        Returns:
            h: [batch, num_nodes, 1] - water level (elevation)
        """
        return depth + self.base_elevation

    def forward(
        self,
        z: torch.Tensor,           # [batch, num_nodes, latent_dim]
        V_prev: torch.Tensor,      # [batch, num_nodes, 1] - previous volume
        h_prev: torch.Tensor,      # [batch, num_nodes, 1] - previous water level
        edge_index: torch.Tensor,  # [2, num_edges]
        source: torch.Tensor,      # [batch, num_nodes, 1] - source (rainfall)
        dt: float = 300.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass: compute new water levels from physics.

        Returns:
            h_new: [batch, num_nodes, 1] - new water level
            V_new: [batch, num_nodes, 1] - new volume
            Q: [batch, num_edges] - flows on edges
        """
        # 1. Compute flows using hydraulic equation
        Q = self.compute_flows(z, h_prev, edge_index)

        # 2. Update volumes via mass balance
        V_new = self.mass_balance_update(V_prev, Q, edge_index, source, dt)

        # 3. Convert volume to depth to water level
        depth = self.volume_to_depth(V_new)
        h_new = self.depth_to_level(depth)

        return h_new, V_new, Q

    def initialize_volume(self, z: torch.Tensor, h_init: torch.Tensor) -> torch.Tensor:
        """Initialize volume from initial water level or latent state.

        Returns:
            V_init: [batch, num_nodes, 1]
        """
        # Option 1: Derive from initial water level
        depth_init = h_init - self.base_elevation
        V_from_h = depth_init * self.node_area.unsqueeze(0).unsqueeze(-1)

        # Option 2: Predict from latent (for flexibility)
        V_from_z = self.volume_init_net(z)

        # Blend: use h_init primarily, z for correction
        V_init = V_from_h + 0.1 * V_from_z

        return V_init.clamp(min=0)


def reparameterize(mu, logvar, temperature=1.0, clamp_value: Optional[float] = None):
    std = torch.exp(0.5 * logvar) * temperature
    eps = torch.randn_like(std)
    z = mu + eps * std
    if clamp_value is not None:
        z = z.clamp(min=-clamp_value, max=clamp_value)
    return z


# ==============================================================================
# Timer-Style Causal Transformer Components
# ==============================================================================

class CausalSelfAttention(nn.Module):
    """Causal (masked) self-attention for autoregressive modeling."""

    def __init__(self, hidden_dim, num_heads=4, dropout=0.1, max_seq_len=512):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Causal mask (lower triangular)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len)).view(1, 1, max_seq_len, max_seq_len)
        )

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, hidden_dim]
        Returns:
            output: [batch, seq_len, hidden_dim]
        """
        B, T, C = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, T, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores with causal mask
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, T, T]
        attn = attn.masked_fill(self.causal_mask[:, :, :T, :T] == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = attn @ v  # [B, num_heads, T, head_dim]
        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.proj(out)
        out = self.dropout(out)

        return out


class TimerBlock(nn.Module):
    """Transformer block for Timer: LayerNorm -> CausalAttn -> LayerNorm -> FFN"""

    def __init__(self, hidden_dim, num_heads=4, ffn_ratio=4, dropout=0.1, max_seq_len=512):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = CausalSelfAttention(hidden_dim, num_heads, dropout, max_seq_len)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ffn_ratio, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


# ==============================================================================
# Timer V4: Bidirectional Attention for Encoding (SCIENTIFIC FIX)
# ==============================================================================
# Key insight: Causal attention is WRONG for encoding prefix sequences.
# For encoding, we want BIDIRECTIONAL attention (like BERT) where every
# position can see all other positions. Causal attention is for GENERATION.
#
# Additional fix: Use MEAN POOLING instead of taking only the last token.
# The GRU baseline works because GRU's hidden state integrates ALL timesteps.
# ==============================================================================


class BidirectionalSelfAttention(nn.Module):
    """Bidirectional (full) self-attention for encoding.

    Unlike CausalSelfAttention, this has NO MASK - every position can attend
    to every other position. This is the correct choice for ENCODING a prefix
    sequence where we want to capture the full context.

    Scientific rationale:
    - Causal attention: position i only sees [1..i] - designed for generation
    - Bidirectional attention: position i sees ALL positions - designed for encoding

    Reference: BERT vs GPT architectural differences.
    """

    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, hidden_dim]
        Returns:
            output: [batch, seq_len, hidden_dim]
        """
        B, T, C = x.shape

        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, T, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]

        # NO CAUSAL MASK - full bidirectional attention
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, num_heads, T, T]
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = attn @ v  # [B, num_heads, T, head_dim]
        out = out.transpose(1, 2).reshape(B, T, C)
        out = self.proj(out)
        out = self.dropout(out)

        return out


class BidirectionalBlock(nn.Module):
    """Transformer block with bidirectional attention for encoding."""

    def __init__(self, hidden_dim, num_heads=4, ffn_ratio=4, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.attn = BidirectionalSelfAttention(hidden_dim, num_heads, dropout)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ffn_ratio, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class TimerPosteriorV4(nn.Module):
    """
    Timer V4: Fixed posterior estimation for z_0 using BIDIRECTIONAL Transformer.

    Key fixes over Timer v3:
    1. BIDIRECTIONAL attention (not causal) - correct for encoding
    2. MEAN POOLING over all positions (not just last token)
    3. LayerNorm preserved (correct, unlike InstanceNorm which removes scale)

    Scientific rationale:
    - For ENCODING a prefix sequence, we want all timesteps to inform each other
    - Causal attention restricts information flow unnecessarily
    - GRU baseline works because its hidden state integrates ALL timesteps
    - Mean pooling matches GRU's behavior of summarizing the full sequence

    Architecture:
        prefix_seq -> Input Projection -> Positional Encoding
        -> N x BidirectionalBlock -> LayerNorm -> MEAN Pooling -> (μ, σ)
    """

    def __init__(self, input_dim, hidden_dim, latent_dim, c_e_dim, spatial_dim,
                 num_nodes, num_layers=4, num_heads=4, dropout=0.1, max_seq_len=64,
                 pooling='mean'):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.pooling = pooling  # 'mean', 'max', or 'attention'

        # Input projection: combine prefix features + spatial + c_e
        self.input_proj = nn.Linear(input_dim + spatial_dim + c_e_dim, hidden_dim)

        # Learnable positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # BIDIRECTIONAL transformer blocks (key fix!)
        self.blocks = nn.ModuleList([
            BidirectionalBlock(hidden_dim, num_heads, ffn_ratio=4, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.ln_out = nn.LayerNorm(hidden_dim)

        # Attention pooling (optional)
        if pooling == 'attention':
            self.pool_query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            nn.init.trunc_normal_(self.pool_query, std=0.02)
            self.pool_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        # Output heads for μ and log(σ²)
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.logvar_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, prefix_seq, spatial_emb, c_e):
        """
        Args:
            prefix_seq: [batch, prefix_len, num_nodes, input_dim] - observed sequence
            spatial_emb: [num_nodes, spatial_dim] - spatial node embeddings
            c_e: [batch, c_e_dim] - event latent

        Returns:
            mu: [batch, num_nodes, latent_dim] - posterior mean
            logvar: [batch, num_nodes, latent_dim] - posterior log variance
        """
        batch, prefix_len, num_nodes, input_dim = prefix_seq.shape

        # Expand spatial and c_e to match prefix sequence dimensions
        spatial_expanded = spatial_emb.unsqueeze(0).unsqueeze(0).expand(batch, prefix_len, -1, -1)
        c_e_expanded = c_e.unsqueeze(1).unsqueeze(2).expand(-1, prefix_len, num_nodes, -1)

        # Concatenate features
        combined = torch.cat([prefix_seq, spatial_expanded, c_e_expanded], dim=-1)

        # Project to hidden dimension
        h = self.input_proj(combined)  # [batch, prefix_len, num_nodes, hidden_dim]

        # Reshape to process each node's sequence through transformer
        # [batch, prefix_len, num_nodes, hidden_dim] -> [batch * num_nodes, prefix_len, hidden_dim]
        h = h.permute(0, 2, 1, 3).reshape(batch * num_nodes, prefix_len, self.hidden_dim)

        # Add positional encoding
        h = h + self.pos_embed[:, :prefix_len, :]

        # Pass through BIDIRECTIONAL transformer blocks
        for block in self.blocks:
            h = block(h)

        h = self.ln_out(h)

        # POOLING: Aggregate information from ALL timesteps (key fix!)
        if self.pooling == 'mean':
            # Mean pooling - simple and effective
            h_pooled = h.mean(dim=1)  # [batch * num_nodes, hidden_dim]
        elif self.pooling == 'max':
            # Max pooling
            h_pooled = h.max(dim=1)[0]  # [batch * num_nodes, hidden_dim]
        elif self.pooling == 'attention':
            # Attention pooling with learnable query
            query = self.pool_query.expand(batch * num_nodes, -1, -1)
            h_pooled, _ = self.pool_attn(query, h, h)  # [batch * num_nodes, 1, hidden_dim]
            h_pooled = h_pooled.squeeze(1)  # [batch * num_nodes, hidden_dim]
        else:
            # Fallback: last position (NOT RECOMMENDED - same as v3)
            h_pooled = h[:, -1, :]

        # Predict posterior parameters
        mu = self.mu_head(h_pooled)  # [batch * num_nodes, latent_dim]
        logvar = self.logvar_head(h_pooled)  # [batch * num_nodes, latent_dim]

        # Clamp logvar for numerical stability
        logvar = torch.clamp(logvar, min=-10, max=2)

        # Reshape to [batch, num_nodes, latent_dim]
        mu = mu.view(batch, num_nodes, self.latent_dim)
        logvar = logvar.view(batch, num_nodes, self.latent_dim)

        return mu, logvar


class TimerTemporalPrior(nn.Module):
    """
    Timer-style temporal prior for latent state transition.

    Takes a history of latent states z_{1:t} and predicts the temporal
    component of the next-step delta. This is combined with the spatial
    (GNN) component in the full transition model.

    Architecture:
        z_{1:t} -> Embedding -> N x TimerBlock -> Output Head -> delta_timer
    """

    def __init__(self, latent_dim, hidden_dim=64, num_layers=4, num_heads=4,
                 dropout=0.1, max_seq_len=512):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Input projection: latent_dim -> hidden_dim
        self.input_proj = nn.Linear(latent_dim, hidden_dim)

        # Positional encoding (learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Timer blocks
        self.blocks = nn.ModuleList([
            TimerBlock(hidden_dim, num_heads, ffn_ratio=4, dropout=dropout, max_seq_len=max_seq_len)
            for _ in range(num_layers)
        ])

        self.ln_out = nn.LayerNorm(hidden_dim)

        # Output head: predict delta in latent space
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z_history):
        """
        Args:
            z_history: [batch, seq_len, num_nodes, latent_dim] - history of latent states

        Returns:
            delta_timer: [batch, num_nodes, latent_dim] - predicted temporal delta for next step
        """
        batch, seq_len, num_nodes, latent_dim = z_history.shape

        # Reshape to process each node's time series
        # [batch, seq_len, num_nodes, latent_dim] -> [batch * num_nodes, seq_len, latent_dim]
        z = z_history.permute(0, 2, 1, 3).reshape(batch * num_nodes, seq_len, latent_dim)

        # Input projection
        h = self.input_proj(z)  # [batch * num_nodes, seq_len, hidden_dim]

        # Add positional encoding
        h = h + self.pos_embed[:, :seq_len, :]

        # Timer blocks
        for block in self.blocks:
            h = block(h)

        h = self.ln_out(h)

        # Take the last timestep's output for next-step prediction
        h_last = h[:, -1, :]  # [batch * num_nodes, hidden_dim]

        # Predict delta
        delta = self.output_head(h_last)  # [batch * num_nodes, latent_dim]

        # Reshape back
        delta = delta.view(batch, num_nodes, latent_dim)

        return delta


class TimerTemporalPriorV5(nn.Module):
    """Successor temporal prior: derivative-aware history encoder.

    Compared to Timer v3 prior, this module:
    - Uses history derivatives (z_t - z_{t-1}) as explicit input.
    - Uses bidirectional encoding over observed history (not generation).
    - Blends last-token and mean-pooled summaries adaptively.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_seq_len: int = 512,
        use_bidirectional: bool = True,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(latent_dim * 2, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        if use_bidirectional:
            self.blocks = nn.ModuleList([
                BidirectionalBlock(hidden_dim, num_heads=num_heads, ffn_ratio=4, dropout=dropout)
                for _ in range(num_layers)
            ])
        else:
            self.blocks = nn.ModuleList([
                TimerBlock(hidden_dim, num_heads=num_heads, ffn_ratio=4, dropout=dropout, max_seq_len=max_seq_len)
                for _ in range(num_layers)
            ])
        self.ln_out = nn.LayerNorm(hidden_dim)

        self.delta_last = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.delta_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.fuse_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Sigmoid(),
        )
        self.magnitude_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh(),
        )

    def forward(self, z_history: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_history: [batch, seq_len, num_nodes, latent_dim]
        Returns:
            delta_timer: [batch, num_nodes, latent_dim]
        """
        batch, seq_len, num_nodes, latent_dim = z_history.shape
        z = z_history.permute(0, 2, 1, 3).reshape(batch * num_nodes, seq_len, latent_dim)

        z_prev = torch.cat([z[:, :1], z[:, :-1]], dim=1)
        dz = z - z_prev
        h = self.input_proj(torch.cat([z, dz], dim=-1))
        h = h + self.pos_embed[:, :seq_len, :]

        for block in self.blocks:
            h = block(h)
        h = self.ln_out(h)

        h_last = h[:, -1, :]
        h_mean = h.mean(dim=1)
        concat = torch.cat([h_last, h_mean], dim=-1)

        delta_last = self.delta_last(h_last)
        delta_mean = self.delta_mean(h_mean)
        gate = self.fuse_gate(concat)
        delta = gate * delta_last + (1.0 - gate) * delta_mean

        # Keep temporal prior magnitude bounded while allowing adaptation.
        scale = 1.0 + 0.25 * self.magnitude_gate(concat)
        delta = delta * scale

        return delta.view(batch, num_nodes, latent_dim)


class TimerEnhancedTransition(nn.Module):
    """
    Latent state transition with Timer temporal prior.

    z_{t+1} = z_t + gate * (delta_spatial + delta_temporal + delta_timer)

    Where:
        - delta_spatial: from GNN (spatial propagation)
        - delta_temporal: from MLP (covariate-based)
        - delta_timer: from Timer (autoregressive temporal patterns)
    """

    def __init__(self, latent_dim, hidden_dim, c_e_dim, control_dim_1d, control_dim_2d,
                 num_gnn_layers=2, dropout=0.1,
                 use_timer=True, timer_layers=4, timer_heads=4, max_seq_len=512,
                 timer_variant: str = 'v3', timer_enable_2d_context: bool = False):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_timer = use_timer
        self.timer_variant = timer_variant
        self.timer_enable_2d_context = bool(timer_enable_2d_context)

        # Existing GNN blocks for spatial
        self.gnn_blocks = nn.ModuleList([
            HeteroGNNBlock(latent_dim, hidden_dim, dropout)
            for _ in range(num_gnn_layers)
        ])

        # Existing MLP for temporal (covariate-based)
        self.temporal_mlp_1d = nn.Sequential(
            nn.Linear(latent_dim + c_e_dim + control_dim_1d, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

        self.temporal_mlp_2d = nn.Sequential(
            nn.Linear(latent_dim + c_e_dim + control_dim_2d, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Timer temporal priors.
        if use_timer:
            if timer_variant in ('v4', 'v5'):
                self.timer_1d = TimerTemporalPriorV5(
                    latent_dim=latent_dim,
                    hidden_dim=hidden_dim,
                    num_layers=timer_layers,
                    num_heads=timer_heads,
                    dropout=dropout,
                    max_seq_len=max_seq_len,
                    use_bidirectional=True,
                )
            else:
                self.timer_1d = TimerTemporalPrior(
                    latent_dim, hidden_dim, timer_layers, timer_heads, dropout, max_seq_len
                )
            self.timer_weight_1d = nn.Parameter(torch.tensor(0.1))
            self.timer_gate_1d = nn.Sequential(
                nn.Linear(latent_dim * 3, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, latent_dim),
                nn.Sigmoid(),
            )
            if self.timer_enable_2d_context:
                self.timer_2d_context = TimerTemporalPriorV5(
                    latent_dim=latent_dim,
                    hidden_dim=hidden_dim,
                    num_layers=max(1, timer_layers // 2),
                    num_heads=timer_heads,
                    dropout=dropout,
                    max_seq_len=max_seq_len,
                    use_bidirectional=True,
                )
                self.timer_2d_proj = nn.Sequential(
                    nn.Linear(latent_dim * 2, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, latent_dim),
                )
                self.timer_weight_2d = nn.Parameter(torch.tensor(0.05))
                self.timer_gate_2d = nn.Sequential(
                    nn.Linear(latent_dim * 3, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, latent_dim),
                    nn.Sigmoid(),
                )
            else:
                self.timer_2d_context = None

        # Gates
        self.gate_1d = nn.Sequential(nn.Linear(latent_dim * 2, latent_dim), nn.Sigmoid())
        self.gate_2d = nn.Sequential(nn.Linear(latent_dim * 2, latent_dim), nn.Sigmoid())

    def forward(self, z_1d, z_2d, edge_index_dict, u_1d, u_2d, c_e,
                z_history_1d=None, z_history_2d=None):
        """
        Args:
            z_1d: [batch, num_1d, latent_dim] - current 1D latent state
            z_2d: [batch, num_2d, latent_dim] - current 2D latent state
            edge_index_dict: graph connectivity
            u_1d: [batch, num_1d, control_dim] - 1D control inputs
            u_2d: [batch, num_2d, control_dim] - 2D control inputs (rainfall)
            c_e: [batch, c_e_dim] - event latent
            z_history_1d: [batch, history_len, num_1d, latent_dim] - optional history for Timer
            z_history_2d: [batch, history_len, num_2d, latent_dim] - optional history for Timer

        Returns:
            z_1d_next, z_2d_next: next latent states
        """
        batch, num_1d, _ = z_1d.shape
        num_2d = z_2d.shape[1]

        # 1. Spatial delta from GNN
        delta_spatial_1d = torch.zeros_like(z_1d)
        delta_spatial_2d = torch.zeros_like(z_2d)

        z_1d_curr = z_1d
        z_2d_curr = z_2d
        for gnn_block in self.gnn_blocks:
            d1d, d2d = gnn_block(z_1d_curr, z_2d_curr, edge_index_dict)
            delta_spatial_1d = delta_spatial_1d + d1d
            delta_spatial_2d = delta_spatial_2d + d2d
            z_1d_curr = z_1d_curr + d1d
            z_2d_curr = z_2d_curr + d2d

        # 2. Temporal delta from MLP (covariate-based)
        c_e_1d = c_e.unsqueeze(1).expand(-1, num_1d, -1)
        c_e_2d = c_e.unsqueeze(1).expand(-1, num_2d, -1)

        if u_1d is not None:
            input_1d = torch.cat([z_1d, c_e_1d, u_1d], dim=-1)
        else:
            input_1d = torch.cat([z_1d, c_e_1d], dim=-1)
            zeros = torch.zeros(batch, num_1d, self.temporal_mlp_1d[0].in_features - input_1d.shape[-1], device=z_1d.device)
            input_1d = torch.cat([input_1d, zeros], dim=-1)

        delta_temporal_1d = self.temporal_mlp_1d(input_1d)

        input_2d = torch.cat([z_2d, c_e_2d, u_2d], dim=-1)
        delta_temporal_2d = self.temporal_mlp_2d(input_2d)

        # 3. Timer delta (autoregressive temporal patterns)
        delta_timer_1d = torch.zeros_like(z_1d)
        delta_timer_2d = torch.zeros_like(z_2d)

        if self.use_timer and self.timer_1d is not None and z_history_1d is not None and z_history_1d.shape[1] > 0:
            delta_timer_1d_raw = self.timer_1d(z_history_1d)
            timer_gate_1d = self.timer_gate_1d(torch.cat([z_1d, delta_temporal_1d, delta_timer_1d_raw], dim=-1))
            delta_timer_1d = timer_gate_1d * delta_timer_1d_raw * self.timer_weight_1d

        if (
            self.use_timer
            and self.timer_enable_2d_context
            and self.timer_2d_context is not None
            and z_history_2d is not None
            and z_history_2d.shape[1] > 0
        ):
            # Low-cost 2D context: summarize 2D history to one latent token per batch,
            # then broadcast learned temporal context back to each 2D node.
            pooled_2d_hist = z_history_2d.mean(dim=2, keepdim=True)  # [B, T, 1, D]
            delta_2d_context = self.timer_2d_context(pooled_2d_hist)[:, 0, :]  # [B, D]
            delta_2d_context = delta_2d_context.unsqueeze(1).expand(-1, num_2d, -1)
            delta_timer_2d_raw = self.timer_2d_proj(torch.cat([z_2d, delta_2d_context], dim=-1))
            timer_gate_2d = self.timer_gate_2d(torch.cat([z_2d, delta_temporal_2d, delta_timer_2d_raw], dim=-1))
            delta_timer_2d = timer_gate_2d * delta_timer_2d_raw * self.timer_weight_2d

        # 4. Combine all deltas
        delta_1d = delta_spatial_1d + delta_temporal_1d + delta_timer_1d
        delta_2d = delta_spatial_2d + delta_temporal_2d + delta_timer_2d

        # 5. Gate and update
        gate_1d = self.gate_1d(torch.cat([z_1d, delta_1d], dim=-1))
        gate_2d = self.gate_2d(torch.cat([z_2d, delta_2d], dim=-1))

        z_1d_next = z_1d + gate_1d * delta_1d
        z_2d_next = z_2d + gate_2d * delta_2d

        return z_1d_next, z_2d_next


# ==============================================================================
# Grassmann Flow Components (Attention-Free Sequence Modeling)
# Based on: "Attention Is Not What You Need: Grassmann Flows"
# https://arxiv.org/abs/2512.19428
# ==============================================================================

class PluckerEncoding(nn.Module):
    """
    Computes Plücker coordinates for 2D subspace spanned by a pair of vectors.

    Given two vectors x_a, x_b in R^r, the Plücker coordinates are:
        p_ij = x_a[i] * x_b[j] - x_a[j] * x_b[i]  for 1 <= i < j <= r

    This gives r(r-1)/2 dimensional output encoding the 2D subspace geometry.
    """

    def __init__(self, reduced_rank: int, normalize: bool = True):
        super().__init__()
        self.reduced_rank = reduced_rank
        self.normalize = normalize
        # Plücker dimension = r(r-1)/2
        self.plucker_dim = reduced_rank * (reduced_rank - 1) // 2

        # Precompute index pairs for efficiency
        pairs = []
        for i in range(reduced_rank):
            for j in range(i + 1, reduced_rank):
                pairs.append((i, j))
        self.register_buffer('pair_i', torch.tensor([p[0] for p in pairs]))
        self.register_buffer('pair_j', torch.tensor([p[1] for p in pairs]))

    def forward(self, x_a: torch.Tensor, x_b: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_a: [..., r] - first vector of the pair
            x_b: [..., r] - second vector of the pair
        Returns:
            plucker: [..., r(r-1)/2] - Plücker coordinates
        """
        # Gather elements for wedge product computation
        # p_ij = x_a[i] * x_b[j] - x_a[j] * x_b[i]
        x_a_i = x_a[..., self.pair_i]  # [..., num_pairs]
        x_a_j = x_a[..., self.pair_j]
        x_b_i = x_b[..., self.pair_i]
        x_b_j = x_b[..., self.pair_j]

        plucker = x_a_i * x_b_j - x_a_j * x_b_i

        if self.normalize:
            # L2 normalize for stability
            plucker = F.normalize(plucker, p=2, dim=-1)

        return plucker


class GrassmannMixingBlock(nn.Module):
    """
    Grassmann Flow mixing block for temporal sequence modeling.

    Captures local geometric structure through multi-scale Plücker encoding:
    1. Linear reduction to rank r
    2. Multi-scale pairing with past timesteps (offsets)
    3. Plücker encoding of 2D subspaces
    4. Aggregate across offsets
    5. Gated fusion with residual

    Complexity: O(L * |Ω| * r²) - linear in sequence length L
    """

    def __init__(
        self,
        hidden_dim: int,
        reduced_rank: int = 12,
        offsets: List[int] = None,
        ffn_ratio: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.reduced_rank = reduced_rank
        self.offsets = offsets if offsets is not None else [1, 2, 4, 8, 16, 32]
        self.num_offsets = len(self.offsets)

        # (1) Linear reduction: hidden_dim -> reduced_rank
        self.W_reduce = nn.Linear(hidden_dim, reduced_rank)

        # (2) Plücker encoding
        self.plucker = PluckerEncoding(reduced_rank, normalize=True)
        plucker_dim = self.plucker.plucker_dim  # r(r-1)/2

        # (3) Projection from Plücker space back to hidden
        # One projection per offset for richer representation
        self.W_project = nn.Linear(plucker_dim * self.num_offsets, hidden_dim)

        # (4) Gated fusion: gate computed from [h_t; g_t]
        self.gate_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid(),
        )

        # (5) FFN (standard transformer-style)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ffn_ratio, hidden_dim),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch, seq_len, hidden_dim] - input sequence
        Returns:
            out: [batch, seq_len, hidden_dim] - output with Grassmann mixing
        """
        batch, seq_len, _ = x.shape

        # (1) Reduce to lower rank for Plücker computation
        x_reduced = self.W_reduce(x)  # [batch, seq_len, reduced_rank]

        # (2) Multi-scale Plücker encoding
        # For each timestep t, pair with (t - offset) for each offset in Ω
        plucker_features = []

        for offset in self.offsets:
            # Create pairs: (x_{t-offset}, x_t) for t >= offset
            # For t < offset, use x_0 as fallback (or zero-pad)
            x_past = torch.zeros_like(x_reduced)
            if offset < seq_len:
                x_past[:, offset:, :] = x_reduced[:, :-offset, :]
                # For early timesteps, repeat first element
                x_past[:, :offset, :] = x_reduced[:, 0:1, :].expand(-1, offset, -1)
            else:
                x_past = x_reduced[:, 0:1, :].expand(-1, seq_len, -1)

            # Compute Plücker coordinates for (x_past, x_current) pairs
            p = self.plucker(x_past, x_reduced)  # [batch, seq_len, plucker_dim]
            plucker_features.append(p)

        # (3) Concatenate across offsets and project
        g = torch.cat(plucker_features, dim=-1)  # [batch, seq_len, plucker_dim * num_offsets]
        g = self.W_project(g)  # [batch, seq_len, hidden_dim]
        g = self.dropout(g)

        # (4) Gated fusion
        x_normed = self.ln1(x)
        gate = self.gate_proj(torch.cat([x_normed, g], dim=-1))  # [batch, seq_len, hidden_dim]
        x = x + gate * g  # Residual with gated Grassmann features

        # (5) FFN with residual
        x = x + self.ffn(self.ln2(x))

        return x


class GrassmannPosterior(nn.Module):
    """
    Grassmann Flow-based posterior encoder for z_0 inference.

    Replaces attention-based (Timer) or RNN-based (GRU) encoders with
    Grassmann mixing blocks that capture local geometric structure in
    flood time series (rise, peak, recession patterns).

    Architecture:
        prefix_seq -> Input Projection -> N x GrassmannMixingBlock
        -> LayerNorm -> Mean Pooling -> (μ, σ)

    Advantages:
    - Linear complexity O(L) vs O(L²) for attention
    - Explicitly models pairwise temporal geometry
    - Multi-scale offsets capture both fast and slow dynamics
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        c_e_dim: int,
        spatial_dim: int,
        num_nodes: int,
        num_layers: int = 4,
        reduced_rank: int = 12,
        offsets: List[int] = None,
        dropout: float = 0.1,
        pooling: str = 'mean',
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_nodes = num_nodes
        self.pooling = pooling

        # Default offsets for flood dynamics: fast (1,2,4) + slow (8,16,32)
        if offsets is None:
            offsets = [1, 2, 4, 8, 16, 32]

        # Input projection: combine prefix features + spatial + c_e
        self.input_proj = nn.Linear(input_dim + spatial_dim + c_e_dim, hidden_dim)

        # Learnable positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, 64, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Grassmann mixing blocks
        self.blocks = nn.ModuleList([
            GrassmannMixingBlock(
                hidden_dim=hidden_dim,
                reduced_rank=reduced_rank,
                offsets=offsets,
                ffn_ratio=4,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.ln_out = nn.LayerNorm(hidden_dim)

        # Attention pooling (optional)
        if pooling == 'attention':
            self.pool_query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            nn.init.trunc_normal_(self.pool_query, std=0.02)
            self.pool_attn = nn.MultiheadAttention(hidden_dim, 4, dropout=dropout, batch_first=True)

        # Output heads for μ and log(σ²)
        self.mu_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.logvar_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, prefix_seq: torch.Tensor, spatial_emb: torch.Tensor, c_e: torch.Tensor):
        """
        Args:
            prefix_seq: [batch, prefix_len, num_nodes, input_dim] - observed sequence
            spatial_emb: [num_nodes, spatial_dim] - spatial node embeddings
            c_e: [batch, c_e_dim] - event latent

        Returns:
            mu: [batch, num_nodes, latent_dim] - posterior mean
            logvar: [batch, num_nodes, latent_dim] - posterior log variance
        """
        batch, prefix_len, num_nodes, input_dim = prefix_seq.shape

        # Expand spatial and c_e to match prefix sequence dimensions
        spatial_expanded = spatial_emb.unsqueeze(0).unsqueeze(0).expand(batch, prefix_len, -1, -1)
        c_e_expanded = c_e.unsqueeze(1).unsqueeze(2).expand(-1, prefix_len, num_nodes, -1)

        # Concatenate features
        combined = torch.cat([prefix_seq, spatial_expanded, c_e_expanded], dim=-1)

        # Reshape: [batch, prefix_len, num_nodes, feat] -> [batch * num_nodes, prefix_len, feat]
        combined = combined.permute(0, 2, 1, 3).reshape(batch * num_nodes, prefix_len, -1)

        # Input projection
        h = self.input_proj(combined)  # [batch * num_nodes, prefix_len, hidden_dim]

        # Add positional encoding
        h = h + self.pos_embed[:, :prefix_len, :]

        # Apply Grassmann mixing blocks
        for block in self.blocks:
            h = block(h)

        h = self.ln_out(h)  # [batch * num_nodes, prefix_len, hidden_dim]

        # Pooling over time dimension
        if self.pooling == 'mean':
            h_pooled = h.mean(dim=1)  # [batch * num_nodes, hidden_dim]
        elif self.pooling == 'max':
            h_pooled = h.max(dim=1)[0]
        elif self.pooling == 'attention':
            query = self.pool_query.expand(batch * num_nodes, -1, -1)
            h_pooled, _ = self.pool_attn(query, h, h)
            h_pooled = h_pooled.squeeze(1)
        else:  # 'last'
            h_pooled = h[:, -1, :]

        # Produce μ and log(σ²)
        mu = self.mu_head(h_pooled)
        logvar = self.logvar_head(h_pooled)

        # Clamp logvar for numerical stability
        logvar = torch.clamp(logvar, min=-10, max=2)

        # Reshape to [batch, num_nodes, latent_dim]
        mu = mu.view(batch, num_nodes, self.latent_dim)
        logvar = logvar.view(batch, num_nodes, self.latent_dim)

        return mu, logvar


class GrassmannTransition(nn.Module):
    """
    Grassmann-enhanced latent transition model.

    Uses Grassmann mixing to compute temporal context from latent history,
    then combines with GNN spatial mixing for the transition dynamics.

    Architecture:
        z_history -> Grassmann mixing -> temporal_context
        z_t + spatial_emb + c_e -> GNN -> spatial_delta
        delta_z = MLP(temporal_context, spatial_delta, control)
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        c_e_dim: int,
        control_dim_1d: int = 1,
        control_dim_2d: int = 1,
        num_gnn_layers: int = 2,
        num_grassmann_layers: int = 2,
        reduced_rank: int = 8,
        offsets: List[int] = None,
        history_len: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.history_len = history_len

        if offsets is None:
            offsets = [1, 2, 4]  # Shorter offsets for transition (recent history)

        # Grassmann temporal mixing for latent history
        self.temporal_proj = nn.Linear(latent_dim, hidden_dim)
        self.grassmann_blocks = nn.ModuleList([
            GrassmannMixingBlock(
                hidden_dim=hidden_dim,
                reduced_rank=reduced_rank,
                offsets=offsets,
                ffn_ratio=2,
                dropout=dropout,
            )
            for _ in range(num_grassmann_layers)
        ])
        self.temporal_out = nn.Linear(hidden_dim, latent_dim)

        # GNN for spatial mixing (unchanged from original)
        self.gnn_layers = nn.ModuleList()
        for i in range(num_gnn_layers):
            in_dim = latent_dim + c_e_dim if i == 0 else hidden_dim
            out_dim = hidden_dim
            self.gnn_layers.append(
                HeteroGNNLayer(
                    in_channels_dict={'1d': in_dim, '2d': in_dim},
                    out_channels=out_dim,
                    edge_types=[('1d', 'pipe', '1d'), ('2d', 'surface', '2d'),
                               ('1d', 'couples', '2d'), ('2d', 'couples', '1d')],
                    use_attention=True,
                    heads=4,
                    dropout=dropout,
                )
            )

        # Delta prediction MLPs
        self.delta_mlp_1d = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim + control_dim_1d, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.delta_mlp_2d = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim + control_dim_2d, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Gating for stability
        self.gate_1d = nn.Sequential(nn.Linear(latent_dim * 2, latent_dim), nn.Sigmoid())
        self.gate_2d = nn.Sequential(nn.Linear(latent_dim * 2, latent_dim), nn.Sigmoid())

    def forward(
        self,
        z_1d: torch.Tensor,
        z_2d: torch.Tensor,
        z_history_1d: torch.Tensor,
        z_history_2d: torch.Tensor,
        c_e: torch.Tensor,
        u_1d: torch.Tensor,
        u_2d: torch.Tensor,
        graph: HeteroData,
    ):
        """
        Args:
            z_1d: [batch, N_1d, latent_dim] - current 1D latent
            z_2d: [batch, N_2d, latent_dim] - current 2D latent
            z_history_1d: [batch, history_len, N_1d, latent_dim] - recent 1D latent history
            z_history_2d: [batch, history_len, N_2d, latent_dim] - recent 2D latent history
            c_e: [batch, c_e_dim] - event latent
            u_1d, u_2d: control inputs
            graph: heterogeneous graph

        Returns:
            z_1d_next, z_2d_next: next latent states
        """
        batch, N_1d, _ = z_1d.shape
        N_2d = z_2d.shape[1]

        # (1) Grassmann temporal mixing on history
        # Reshape: [batch, history, N, latent] -> [batch*N, history, latent]
        h_1d = z_history_1d.permute(0, 2, 1, 3).reshape(batch * N_1d, -1, self.latent_dim)
        h_2d = z_history_2d.permute(0, 2, 1, 3).reshape(batch * N_2d, -1, self.latent_dim)

        h_1d = self.temporal_proj(h_1d)
        h_2d = self.temporal_proj(h_2d)

        for block in self.grassmann_blocks:
            h_1d = block(h_1d)
            h_2d = block(h_2d)

        # Take last position as temporal context
        temp_ctx_1d = self.temporal_out(h_1d[:, -1, :]).view(batch, N_1d, -1)
        temp_ctx_2d = self.temporal_out(h_2d[:, -1, :]).view(batch, N_2d, -1)

        # (2) GNN spatial mixing
        c_e_1d = c_e.unsqueeze(1).expand(-1, N_1d, -1)
        c_e_2d = c_e.unsqueeze(1).expand(-1, N_2d, -1)

        h_1d_gnn = torch.cat([z_1d, c_e_1d], dim=-1)
        h_2d_gnn = torch.cat([z_2d, c_e_2d], dim=-1)

        for gnn in self.gnn_layers:
            h_1d_gnn, h_2d_gnn = gnn(h_1d_gnn, h_2d_gnn, graph)

        # (3) Combine temporal + spatial + control for delta prediction
        delta_in_1d = torch.cat([h_1d_gnn, temp_ctx_1d, u_1d], dim=-1)
        delta_in_2d = torch.cat([h_2d_gnn, temp_ctx_2d, u_2d], dim=-1)

        delta_1d = self.delta_mlp_1d(delta_in_1d)
        delta_2d = self.delta_mlp_2d(delta_in_2d)

        # (4) Gated residual update
        gate_1d = self.gate_1d(torch.cat([z_1d, delta_1d], dim=-1))
        gate_2d = self.gate_2d(torch.cat([z_2d, delta_2d], dim=-1))

        z_1d_next = z_1d + gate_1d * delta_1d
        z_2d_next = z_2d + gate_2d * delta_2d

        return z_1d_next, z_2d_next


class VGSSM(nn.Module):
    """Variational Graph State-Space Model for Urban Flood Prediction.

    Physics-Informed Version with:
    - Delta prediction for stable long-horizon rollouts
    - Edge flow prediction for mass conservation
    - Support for physics loss computation
    """

    def __init__(
        self,
        static_1d_dim, static_2d_dim,
        dynamic_1d_dim, dynamic_2d_dim,
        hidden_dim=64, latent_dim=32, event_latent_dim=16,
        num_gnn_layers=3, num_transition_gnn_layers=2, num_heads=4,
        prediction_horizon=90, use_event_latent=True, dropout=0.1,
        use_delta_prediction=False, use_physics_loss=False,
        physics_subsample_rate=5,  # Memory optimization: only compute physics loss every N timesteps
        # Timer configuration
        use_timer=False, timer_layers=4, timer_heads=4, timer_history_len=10,
        # Timer V4: Scientific fix with bidirectional attention and mean pooling
        use_timer_v4=False, timer_v4_pooling='mean',  # 'mean', 'max', 'attention'
        timer_transition_variant='auto',  # auto | v3 | v5
        timer_enable_2d_context: Optional[bool] = None,
        # Physics-constrained transition: blends neural + physics updates
        use_physics_transition=False,
        # Grassmann Flow configuration (attention-free sequence modeling)
        use_grassmann=False,  # Use Grassmann posterior instead of Timer/GRU
        grassmann_layers=4,   # Number of Grassmann mixing blocks
        grassmann_rank=12,    # Reduced rank for Plücker encoding (r -> r(r-1)/2 dim)
        grassmann_offsets=None,  # Multi-scale offsets, default [1,2,4,8,16,32]
        # Output bounds for water level predictions (prevents invalid values)
        output_bounds_1d: Tuple[float, float] = None,  # (min, max) for 1D nodes
        output_bounds_2d: Tuple[float, float] = None,  # (min, max) for 2D nodes
        # Baseline residual mode: predict deviations from per-node mean
        use_baseline_residual: bool = False,
        num_1d_nodes: int = None,  # Required if use_baseline_residual
        num_2d_nodes: int = None,  # Required if use_baseline_residual
        # Sigmoid bounds: h = min + sigmoid(x) * (max - min)
        use_sigmoid_bounds: bool = False,
        # Physics decoder: derive h from mass conservation (K -> Q -> V -> h)
        use_physics_decoder: bool = False,
        node_areas_1d: torch.Tensor = None,  # Storage area per 1D node [m²]
        node_areas_2d: torch.Tensor = None,  # Storage area per 2D node [m²]
        physics_dt: float = 300.0,           # Time step for physics [seconds]
        latent_sample_temperature: float = 1.0,  # Scale posterior sampling noise
        latent_state_clip: float = 10.0,  # Clamp latent state to avoid rollout blow-up
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.event_latent_dim = event_latent_dim if use_event_latent else 0
        self.use_event_latent = use_event_latent
        self.prediction_horizon = prediction_horizon
        self.dynamic_1d_dim = dynamic_1d_dim
        self.dynamic_2d_dim = dynamic_2d_dim
        self.use_delta_prediction = use_delta_prediction
        self.use_physics_loss = use_physics_loss
        self.physics_subsample_rate = physics_subsample_rate  # Reduces memory by this factor
        self.use_timer = bool(use_timer or use_timer_v4)
        self.use_timer_v4 = use_timer_v4
        self.timer_history_len = timer_history_len
        if timer_transition_variant == 'auto':
            self.timer_transition_variant = 'v5' if use_timer_v4 else 'v3'
        else:
            self.timer_transition_variant = str(timer_transition_variant)
        if timer_enable_2d_context is None:
            self.timer_enable_2d_context = bool(use_timer_v4)
        else:
            self.timer_enable_2d_context = bool(timer_enable_2d_context)
        self.use_physics_transition = use_physics_transition
        self.use_grassmann = use_grassmann
        self.latent_sample_temperature = float(max(latent_sample_temperature, 0.0))
        self.latent_state_clip = float(latent_state_clip) if latent_state_clip and latent_state_clip > 0 else None

        self.spatial_encoder = SpatialEncoder(
            static_1d_dim, static_2d_dim,
            hidden_channels=hidden_dim, num_layers=num_gnn_layers,
            use_attention=True, dropout=dropout,
        )

        if use_event_latent:
            self.event_encoder = EventLatentEncoderTFT(
                input_dim=dynamic_1d_dim + dynamic_2d_dim,
                hidden_dim=hidden_dim, latent_dim=event_latent_dim,
                num_heads=num_heads, dropout=dropout,
            )
            self.event_prior_mean = nn.Parameter(torch.zeros(event_latent_dim))
            self.event_prior_logvar = nn.Parameter(torch.zeros(event_latent_dim))

        # Select posterior encoder based on configuration
        # Priority: Grassmann > Timer V4 > Timer V3 > GRU baseline
        if use_grassmann:
            # Grassmann Flow: attention-free with multi-scale Plücker encoding
            # Captures local geometric structure (rise, peak, recession)
            # Linear complexity O(L) vs O(L²) for attention
            offsets = grassmann_offsets if grassmann_offsets else [1, 2, 4, 8, 16, 32]
            self.z0_encoder_1d = GrassmannPosterior(
                input_dim=dynamic_1d_dim, hidden_dim=hidden_dim, latent_dim=latent_dim,
                c_e_dim=self.event_latent_dim, spatial_dim=hidden_dim, num_nodes=1,
                num_layers=grassmann_layers, reduced_rank=grassmann_rank,
                offsets=offsets, dropout=dropout, pooling='mean',
            )
            # 2D also uses Grassmann (linear complexity makes it feasible)
            self.z0_encoder_2d = GrassmannPosterior(
                input_dim=dynamic_2d_dim, hidden_dim=hidden_dim, latent_dim=latent_dim,
                c_e_dim=self.event_latent_dim, spatial_dim=hidden_dim, num_nodes=1,
                num_layers=grassmann_layers, reduced_rank=grassmann_rank,
                offsets=offsets, dropout=dropout, pooling='mean',
            )
        elif use_timer_v4:
            # Timer V4: SCIENTIFIC FIX - bidirectional attention + mean pooling
            # Fixes the fundamental issues with Timer v3:
            # 1. Uses BIDIRECTIONAL attention (not causal) - correct for encoding
            # 2. Uses MEAN POOLING (not last token) - matches GRU's behavior
            self.z0_encoder_1d = TimerPosteriorV4(
                input_dim=dynamic_1d_dim, hidden_dim=hidden_dim, latent_dim=latent_dim,
                c_e_dim=self.event_latent_dim, spatial_dim=hidden_dim, num_nodes=1,
                num_layers=timer_layers, num_heads=timer_heads, dropout=dropout, max_seq_len=64,
                pooling=timer_v4_pooling,
            )
            # 2D uses GRU (attention too memory-intensive for 3716 nodes)
            self.z0_encoder_2d = LatentInferenceNet(
                input_dim=dynamic_2d_dim, hidden_dim=hidden_dim, latent_dim=latent_dim,
                c_e_dim=self.event_latent_dim, spatial_dim=hidden_dim, num_nodes=1, dropout=dropout,
            )
        elif use_timer:
            # Timer V3: Original (DEPRECATED - has fundamental issues)
            # Issues: causal attention + last token only = information bottleneck
            self.z0_encoder_1d = TimerPosterior(
                input_dim=dynamic_1d_dim, hidden_dim=hidden_dim, latent_dim=latent_dim,
                c_e_dim=self.event_latent_dim, spatial_dim=hidden_dim, num_nodes=1,
                num_layers=timer_layers, num_heads=timer_heads, dropout=dropout, max_seq_len=64,
            )
            self.z0_encoder_2d = LatentInferenceNet(
                input_dim=dynamic_2d_dim, hidden_dim=hidden_dim, latent_dim=latent_dim,
                c_e_dim=self.event_latent_dim, spatial_dim=hidden_dim, num_nodes=1, dropout=dropout,
            )
        else:
            # GRU-based (baseline - works well)
            self.z0_encoder_1d = LatentInferenceNet(
                input_dim=dynamic_1d_dim, hidden_dim=hidden_dim, latent_dim=latent_dim,
                c_e_dim=self.event_latent_dim, spatial_dim=hidden_dim, num_nodes=1, dropout=dropout,
            )
            self.z0_encoder_2d = LatentInferenceNet(
                input_dim=dynamic_2d_dim, hidden_dim=hidden_dim, latent_dim=latent_dim,
                c_e_dim=self.event_latent_dim, spatial_dim=hidden_dim, num_nodes=1, dropout=dropout,
            )

        self.z0_prior_mean = nn.Parameter(torch.zeros(latent_dim))
        self.z0_prior_logvar = nn.Parameter(torch.zeros(latent_dim))

        control_dim_1d = 1
        control_dim_2d = 1

        # Select transition model based on configuration
        # Priority: PhysicsConstrainedTransition > TimerEnhanced > Standard
        if use_physics_transition:
            # Physics-Constrained Transition: blends neural + physics updates
            # This integrates mass conservation directly into the transition dynamics
            self.transition = PhysicsConstrainedTransition(
                latent_dim=latent_dim, hidden_dim=hidden_dim, c_e_dim=self.event_latent_dim,
                control_dim_1d=control_dim_1d, control_dim_2d=control_dim_2d,
                num_gnn_layers=num_transition_gnn_layers, dropout=dropout,
            )
        elif self.use_timer:
            self.transition = TimerEnhancedTransition(
                latent_dim=latent_dim, hidden_dim=hidden_dim, c_e_dim=self.event_latent_dim,
                control_dim_1d=control_dim_1d, control_dim_2d=control_dim_2d,
                num_gnn_layers=num_transition_gnn_layers, dropout=dropout,
                use_timer=True, timer_layers=timer_layers, timer_heads=timer_heads,
                max_seq_len=timer_history_len + prediction_horizon,
                timer_variant=self.timer_transition_variant,
                timer_enable_2d_context=self.timer_enable_2d_context,
            )
        else:
            self.transition = LatentTransition(
                latent_dim=latent_dim, hidden_dim=hidden_dim, c_e_dim=self.event_latent_dim,
                control_dim_1d=control_dim_1d, control_dim_2d=control_dim_2d,
                num_gnn_layers=num_transition_gnn_layers, dropout=dropout,
            )
        self.transition_uses_timer_history = isinstance(self.transition, TimerEnhancedTransition)

        self.output_dim_1d = 1
        self.output_dim_2d = 1

        # Store output bounds
        self.output_bounds_1d = output_bounds_1d
        self.output_bounds_2d = output_bounds_2d

        self.use_baseline_residual = use_baseline_residual
        self.use_sigmoid_bounds = use_sigmoid_bounds
        self.use_physics_decoder = use_physics_decoder
        self.physics_dt = physics_dt

        if use_physics_decoder:
            # Physics-based decoder: K -> Q -> V -> h (mass-conserving)
            # Base elevation from output bounds
            base_elev_1d = output_bounds_1d[0] if output_bounds_1d else 0.0
            base_elev_2d = output_bounds_2d[0] if output_bounds_2d else 0.0
            max_depth_1d = (output_bounds_1d[1] - output_bounds_1d[0]) if output_bounds_1d else 50.0
            max_depth_2d = (output_bounds_2d[1] - output_bounds_2d[0]) if output_bounds_2d else 50.0

            self.physics_decoder_1d = PhysicsDecoder(
                latent_dim=latent_dim, hidden_dim=hidden_dim, num_nodes=num_1d_nodes,
                node_areas=node_areas_1d, min_depth=0.0, max_depth=max_depth_1d,
                base_elevation=base_elev_1d, dropout=dropout,
            )
            self.physics_decoder_2d = PhysicsDecoder(
                latent_dim=latent_dim, hidden_dim=hidden_dim, num_nodes=num_2d_nodes,
                node_areas=node_areas_2d, min_depth=0.0, max_depth=max_depth_2d,
                base_elevation=base_elev_2d, dropout=dropout,
            )
            # Dummy decoders for compatibility (not used when physics_decoder is active)
            self.decoder_1d = None
            self.decoder_2d = None
        else:
            # Standard LatentDecoder (direct h prediction)
            self.decoder_1d = LatentDecoder(
                latent_dim, hidden_dim, hidden_dim, self.output_dim_1d,
                delta_mode=use_delta_prediction, dropout=dropout,
                output_min=output_bounds_1d[0] if output_bounds_1d else None,
                output_max=output_bounds_1d[1] if output_bounds_1d else None,
                use_baseline_residual=use_baseline_residual,
                num_nodes=num_1d_nodes,
                use_sigmoid_bounds=use_sigmoid_bounds,
            )
            self.decoder_2d = LatentDecoder(
                latent_dim, hidden_dim, hidden_dim, self.output_dim_2d,
                delta_mode=use_delta_prediction, dropout=dropout,
                output_min=output_bounds_2d[0] if output_bounds_2d else None,
                output_max=output_bounds_2d[1] if output_bounds_2d else None,
                use_baseline_residual=use_baseline_residual,
                num_nodes=num_2d_nodes,
                use_sigmoid_bounds=use_sigmoid_bounds,
            )
            self.physics_decoder_1d = None
            self.physics_decoder_2d = None

        # Physics-informed components
        if use_physics_loss:
            # Physics-based flow computation (derives flow from water level differences)
            # This is the proper PINN approach: Q = f(Δh), not Q = NN(z)
            self.physics_flow = PhysicsBasedFlow(hidden_dim=hidden_dim, max_flow=10.0)
            # Simplified physics loss for volume balance and smoothness
            self.simplified_physics = SimplifiedPhysicsLoss(dt_seconds=300.0)
            # Keep old EdgeFlowHead for backward compatibility (not used in v6+)
            self.flow_head_1d = EdgeFlowHead(latent_dim, hidden_dim, dropout=dropout)
            self.flow_head_2d = EdgeFlowHead(latent_dim, hidden_dim, dropout=dropout)
            self.flow_head_coupling = EdgeFlowHead(latent_dim, hidden_dim, dropout=dropout)

    def encode_spatial(self, graph):
        return self.spatial_encoder(graph)

    def encode_event_latent(self, prefix_1d, prefix_2d, deterministic=False):
        if not self.use_event_latent:
            batch = prefix_1d.shape[0]
            device = prefix_1d.device
            zeros = torch.zeros(batch, 0, device=device)
            return zeros, zeros, zeros

        pooled_1d = prefix_1d.mean(dim=2)
        pooled_2d = prefix_2d.mean(dim=2)
        combined = torch.cat([pooled_1d, pooled_2d], dim=-1).unsqueeze(2)

        mean, logvar = self.event_encoder(combined)
        if deterministic or self.latent_sample_temperature <= 0:
            c_e = mean
        else:
            c_e = self.event_encoder.sample(mean, logvar, temperature=self.latent_sample_temperature)
        if self.latent_state_clip is not None:
            c_e = c_e.clamp(min=-self.latent_state_clip, max=self.latent_state_clip)

        return c_e, mean, logvar

    def compute_edge_flows(self, z_1d: torch.Tensor, z_2d: torch.Tensor,
                           edge_index_dict: Dict[Tuple, torch.Tensor]
                           ) -> Dict[str, torch.Tensor]:
        """Compute flows on all edges using latent states.

        Returns:
            flows: Dict mapping edge type name to flow tensor [batch, num_edges]
        """
        if not self.use_physics_loss:
            return {}

        batch = z_1d.shape[0]
        flows = {}

        # Process each batch item separately
        for edge_type, edge_index in edge_index_dict.items():
            src_type, edge_name, dst_type = edge_type
            num_edges = edge_index.shape[1]

            # Select appropriate flow head
            if edge_name == 'pipe':
                flow_head = self.flow_head_1d
                z_src_all = z_1d
                z_dst_all = z_1d
            elif edge_name == 'surface':
                flow_head = self.flow_head_2d
                z_src_all = z_2d
                z_dst_all = z_2d
            else:  # coupling edges
                flow_head = self.flow_head_coupling
                z_src_all = z_1d if src_type == '1d' else z_2d
                z_dst_all = z_2d if dst_type == '2d' else z_1d

            batch_flows = []
            for b in range(batch):
                z_src = z_src_all[b, edge_index[0]]  # [num_edges, latent_dim]
                z_dst = z_dst_all[b, edge_index[1]]

                flow = flow_head(z_src, z_dst)  # [num_edges, 1]
                batch_flows.append(flow.squeeze(-1))

            flows[edge_name] = torch.stack(batch_flows, dim=0)  # [batch, num_edges]

        return flows

    def aggregate_flows_to_nodes(self, flows: Dict[str, torch.Tensor],
                                  edge_index_dict: Dict[Tuple, torch.Tensor],
                                  num_1d: int, num_2d: int, batch: int, device
                                  ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Aggregate edge flows to node-level inflows and outflows.

        Returns:
            inflows_1d, outflows_1d, inflows_2d, outflows_2d: [batch, num_nodes]
        """
        inflows_1d = torch.zeros(batch, num_1d, device=device)
        outflows_1d = torch.zeros(batch, num_1d, device=device)
        inflows_2d = torch.zeros(batch, num_2d, device=device)
        outflows_2d = torch.zeros(batch, num_2d, device=device)

        for edge_type, edge_index in edge_index_dict.items():
            src_type, edge_name, dst_type = edge_type

            if edge_name not in flows:
                continue

            edge_flows = flows[edge_name]  # [batch, num_edges]
            src_idx = edge_index[0]
            dst_idx = edge_index[1]

            for b in range(batch):
                flow = edge_flows[b]  # [num_edges]

                # Positive flow: src -> dst
                # Negative flow: dst -> src
                pos_flow = F.relu(flow)
                neg_flow = F.relu(-flow)

                if src_type == '1d':
                    outflows_1d[b].scatter_add_(0, src_idx, pos_flow)
                    inflows_1d[b].scatter_add_(0, src_idx, neg_flow)
                else:
                    outflows_2d[b].scatter_add_(0, src_idx, pos_flow)
                    inflows_2d[b].scatter_add_(0, src_idx, neg_flow)

                if dst_type == '1d':
                    inflows_1d[b].scatter_add_(0, dst_idx, pos_flow)
                    outflows_1d[b].scatter_add_(0, dst_idx, neg_flow)
                else:
                    inflows_2d[b].scatter_add_(0, dst_idx, pos_flow)
                    outflows_2d[b].scatter_add_(0, dst_idx, neg_flow)

        return inflows_1d, outflows_1d, inflows_2d, outflows_2d

    def compute_physics_flows(self, h_1d: torch.Tensor, h_2d: torch.Tensor,
                               edge_index_dict: Dict[Tuple, torch.Tensor]
                               ) -> Dict[str, torch.Tensor]:
        """Compute flows from WATER LEVEL differences using physical equations.

        This is the proper physics-informed approach where Q = f(Δh).
        Flow is driven by head difference, not by arbitrary neural network on latents.

        Args:
            h_1d: Water levels at 1D nodes [batch, num_1d]
            h_2d: Water levels at 2D nodes [batch, num_2d]
            edge_index_dict: Edge indices for each edge type

        Returns:
            flows: Dict mapping edge type name to flow tensor [batch, num_edges]
        """
        if not self.use_physics_loss or not hasattr(self, 'physics_flow'):
            return {}

        batch = h_1d.shape[0]
        flows = {}

        for edge_type, edge_index in edge_index_dict.items():
            src_type, edge_name, dst_type = edge_type
            src_idx = edge_index[0]
            dst_idx = edge_index[1]

            # Select water levels based on node types
            if src_type == '1d':
                h_src_all = h_1d
            else:
                h_src_all = h_2d

            if dst_type == '1d':
                h_dst_all = h_1d
            else:
                h_dst_all = h_2d

            batch_flows = []
            for b in range(batch):
                h_src = h_src_all[b, src_idx]  # [num_edges]
                h_dst = h_dst_all[b, dst_idx]

                # Compute flow from water level difference using physics equations
                flow = self.physics_flow(h_src, h_dst, edge_name)
                batch_flows.append(flow)

            flows[edge_name] = torch.stack(batch_flows, dim=0)  # [batch, num_edges]

        return flows

    def forward(self, graph, input_1d, input_2d, prefix_len=10, future_rainfall=None,
                future_inlet_flow=None, c_e_override=None, z0_1d_override=None, z0_2d_override=None,
                rollout_len=None, return_flows=False, h0_1d=None, h0_2d=None,
                deterministic_latent=False, return_final_state: bool = False):
        """
        Args:
            graph: HeteroData graph
            input_1d: [batch, seq, num_1d, dynamic_1d_dim]
            input_2d: [batch, seq, num_2d, dynamic_2d_dim]
            prefix_len: Number of warmup timesteps
            future_rainfall: [batch, horizon, num_2d, 1]
            future_inlet_flow: [batch, horizon, num_1d, 1]
            c_e_override: Override event latent
            z0_1d_override: Override initial 1D latent
            z0_2d_override: Override initial 2D latent
            rollout_len: Override prediction horizon (for curriculum)
            return_flows: If True, return edge flows for physics loss
            h0_1d: Initial water level for 1D nodes (for delta mode)
            h0_2d: Initial water level for 2D nodes (for delta mode)

        Returns:
            outputs: Dict with predictions and latent variables
        """
        batch, seq_len, num_1d_nodes, _ = input_1d.shape
        num_2d_nodes = input_2d.shape[2]
        device = input_1d.device

        spatial_1d, spatial_2d = self.encode_spatial(graph)

        edge_index_dict = {}
        for edge_type in self.transition.gnn_blocks[0].edge_types:
            if edge_type in graph.edge_types:
                edge_index_dict[edge_type] = graph[edge_type].edge_index

        if c_e_override is not None:
            c_e = c_e_override
            c_e_mean = c_e_override
            c_e_logvar = torch.zeros_like(c_e_override)
        else:
            prefix_1d = input_1d[:, :prefix_len]
            prefix_2d = input_2d[:, :prefix_len]
            c_e, c_e_mean, c_e_logvar = self.encode_event_latent(
                prefix_1d, prefix_2d, deterministic=deterministic_latent
            )

        if z0_1d_override is not None:
            z0_mu_1d = z0_1d_override
            z0_logvar_1d = torch.zeros_like(z0_1d_override)
            z_1d = z0_1d_override
        else:
            prefix_1d = input_1d[:, :prefix_len]
            z0_mu_1d, z0_logvar_1d = self.z0_encoder_1d(prefix_1d, spatial_1d, c_e)
            if deterministic_latent or self.latent_sample_temperature <= 0:
                z_1d = z0_mu_1d
            else:
                z_1d = reparameterize(
                    z0_mu_1d,
                    z0_logvar_1d,
                    temperature=self.latent_sample_temperature,
                    clamp_value=self.latent_state_clip,
                )

        if z0_2d_override is not None:
            z0_mu_2d = z0_2d_override
            z0_logvar_2d = torch.zeros_like(z0_2d_override)
            z_2d = z0_2d_override
        else:
            prefix_2d = input_2d[:, :prefix_len]
            z0_mu_2d, z0_logvar_2d = self.z0_encoder_2d(prefix_2d, spatial_2d, c_e)
            if deterministic_latent or self.latent_sample_temperature <= 0:
                z_2d = z0_mu_2d
            else:
                z_2d = reparameterize(
                    z0_mu_2d,
                    z0_logvar_2d,
                    temperature=self.latent_sample_temperature,
                    clamp_value=self.latent_state_clip,
                )

        # Determine horizon
        if rollout_len is not None:
            horizon = rollout_len
        elif future_rainfall is not None:
            horizon = min(self.prediction_horizon, future_rainfall.shape[1])
        else:
            horizon = self.prediction_horizon

        # Initialize h_prev for delta mode or physics decoder
        if self.use_delta_prediction or self.use_physics_decoder:
            if h0_1d is not None:
                h_prev_1d = h0_1d
            else:
                # Use last input water level
                h_prev_1d = input_1d[:, -1, :, 0:1]  # [batch, num_1d, 1]

            if h0_2d is not None:
                h_prev_2d = h0_2d
            else:
                h_prev_2d = input_2d[:, -1, :, 1:2]  # water_level is index 1
        else:
            h_prev_1d = None
            h_prev_2d = None

        # Initialize volumes for physics decoder
        if self.use_physics_decoder:
            V_prev_1d = self.physics_decoder_1d.initialize_volume(z_1d, h_prev_1d)
            V_prev_2d = self.physics_decoder_2d.initialize_volume(z_2d, h_prev_2d)
            # Get edge indices for 1D and 2D subgraphs
            edge_index_1d = edge_index_dict.get(('1d', 'pipe', '1d'), None)
            edge_index_2d = edge_index_dict.get(('2d', 'surface', '2d'), None)
        else:
            V_prev_1d = None
            V_prev_2d = None
            edge_index_1d = None
            edge_index_2d = None

        preds_1d = []
        preds_2d = []
        deltas_1d = []
        deltas_2d = []
        flows_1d_list = []
        flows_2d_list = []
        # Memory optimization: only store flows for subsampled timesteps
        all_flows = {} if return_flows else None  # Changed to dict with timestep keys
        physics_subsample_rate = getattr(self, 'physics_subsample_rate', 5)

        # Timer history buffers (for Timer-enhanced transition)
        if self.transition_uses_timer_history:
            timer_history_len = getattr(self, 'timer_history_len', 10)
            # Initialize with repeated initial state
            z_history_1d = z_1d.unsqueeze(1).expand(-1, timer_history_len, -1, -1).clone()
            z_history_2d = z_2d.unsqueeze(1).expand(-1, timer_history_len, -1, -1).clone()
        else:
            z_history_1d = None
            z_history_2d = None

        for t in range(horizon):
            u_2d = future_rainfall[:, t] if future_rainfall is not None and t < future_rainfall.shape[1] else torch.zeros(batch, num_2d_nodes, 1, device=device)
            u_1d = future_inlet_flow[:, t] if future_inlet_flow is not None and t < future_inlet_flow.shape[1] else None

            # Call transition with Timer history if enabled
            if self.transition_uses_timer_history:
                z_1d, z_2d = self.transition(
                    z_1d, z_2d, edge_index_dict, u_1d, u_2d, c_e,
                    z_history_1d=z_history_1d, z_history_2d=z_history_2d
                )
                # Update history (shift and append)
                z_history_1d = torch.cat([z_history_1d[:, 1:], z_1d.unsqueeze(1)], dim=1)
                z_history_2d = torch.cat([z_history_2d[:, 1:], z_2d.unsqueeze(1)], dim=1)
            else:
                z_1d, z_2d = self.transition(z_1d, z_2d, edge_index_dict, u_1d, u_2d, c_e)

            if self.latent_state_clip is not None:
                z_1d = z_1d.clamp(min=-self.latent_state_clip, max=self.latent_state_clip)
                z_2d = z_2d.clamp(min=-self.latent_state_clip, max=self.latent_state_clip)

            # Decode with physics decoder or standard decoder
            if self.use_physics_decoder:
                # Physics decoder: K -> Q -> V -> h (mass-conserving)
                # Source term is rainfall for 2D, inlet flow for 1D
                source_1d = u_1d if u_1d is not None else torch.zeros_like(h_prev_1d)
                source_2d = u_2d

                if edge_index_1d is not None:
                    pred_1d, V_prev_1d, Q_1d = self.physics_decoder_1d(
                        z_1d, V_prev_1d, h_prev_1d, edge_index_1d, source_1d, self.physics_dt
                    )
                    flows_1d_list.append(Q_1d)
                else:
                    # No 1D edges - just use volume update without flow
                    pred_1d = h_prev_1d  # Fallback

                if edge_index_2d is not None:
                    pred_2d, V_prev_2d, Q_2d = self.physics_decoder_2d(
                        z_2d, V_prev_2d, h_prev_2d, edge_index_2d, source_2d, self.physics_dt
                    )
                    flows_2d_list.append(Q_2d)
                else:
                    pred_2d = h_prev_2d  # Fallback

                delta_1d = pred_1d - h_prev_1d
                delta_2d = pred_2d - h_prev_2d
                h_prev_1d = pred_1d
                h_prev_2d = pred_2d
            else:
                # Standard decoder: direct h prediction
                pred_1d, delta_1d = self.decoder_1d(z_1d, spatial_1d, h_prev_1d)
                pred_2d, delta_2d = self.decoder_2d(z_2d, spatial_2d, h_prev_2d)

                # Update h_prev for next step
                if self.use_delta_prediction:
                    h_prev_1d = pred_1d
                    h_prev_2d = pred_2d

            preds_1d.append(pred_1d)
            preds_2d.append(pred_2d)
            deltas_1d.append(delta_1d)
            deltas_2d.append(delta_2d)

            # Memory optimization: only compute flows for subsampled timesteps
            # This reduces memory by physics_subsample_rate (e.g., 5x)
            if return_flows and self.use_physics_loss:
                if t % physics_subsample_rate == 0 or t == horizon - 1:
                    flows_t = self.compute_edge_flows(z_1d, z_2d, edge_index_dict)
                    all_flows[t] = flows_t

        pred_1d = torch.stack(preds_1d, dim=1)
        pred_2d = torch.stack(preds_2d, dim=1)
        delta_1d = torch.stack(deltas_1d, dim=1) if self.use_delta_prediction else None
        delta_2d = torch.stack(deltas_2d, dim=1) if self.use_delta_prediction else None

        outputs = {
            'pred_1d': pred_1d,
            'pred_2d': pred_2d,
            'c_e': c_e,
            'c_e_mean': c_e_mean,
            'c_e_logvar': c_e_logvar,
            'z0_mu_1d': z0_mu_1d,
            'z0_logvar_1d': z0_logvar_1d,
            'z0_mu_2d': z0_mu_2d,
            'z0_logvar_2d': z0_logvar_2d,
        }

        if self.use_delta_prediction:
            outputs['delta_1d'] = delta_1d
            outputs['delta_2d'] = delta_2d

        # Always include edge_index_dict for physics loss (needed for smoothness)
        if self.use_physics_loss:
            outputs['edge_index_dict'] = edge_index_dict

        if return_flows and all_flows:
            outputs['flows'] = all_flows  # Now a dict with timestep keys
            outputs['physics_subsample_rate'] = physics_subsample_rate

        if return_final_state:
            outputs['zT_1d'] = z_1d
            outputs['zT_2d'] = z_2d
            if self.use_delta_prediction or self.use_physics_decoder:
                outputs['hT_1d'] = h_prev_1d
                outputs['hT_2d'] = h_prev_2d

        return outputs


# =============================================================================
# VGSSM TRAINER
# =============================================================================

class CurriculumCallback(Callback):
    """PyTorch Lightning callback for curriculum learning."""

    def __init__(self, scheduler: CurriculumScheduler):
        self.scheduler = scheduler

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        val_loss = trainer.callback_metrics.get('val/std_rmse_curr')
        if val_loss is None:
            val_loss = trainer.callback_metrics.get('val/std_rmse')
        if val_loss is not None:
            self.scheduler.step(val_loss.item())

    def state_dict(self):
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict)


class CurriculumAwareEarlyStopping(EarlyStopping):
    """Apply early stopping only once curriculum reaches final rollout stage."""

    def _should_skip(self, trainer, pl_module) -> bool:
        if trainer.sanity_checking:
            return True
        if (
            getattr(pl_module, 'use_curriculum', False)
            and getattr(pl_module, 'curriculum_scheduler', None) is not None
            and not pl_module.curriculum_scheduler.is_final_stage
        ):
            return True
        return False

    def on_validation_end(self, trainer, pl_module):
        if self._should_skip(trainer, pl_module):
            return
        super().on_validation_end(trainer, pl_module)

    def on_train_epoch_end(self, trainer, pl_module):
        if self._should_skip(trainer, pl_module):
            return
        super().on_train_epoch_end(trainer, pl_module)


class VGSSMTrainer(pl.LightningModule):
    """PyTorch Lightning module for training Physics-Informed VGSSM."""

    def __init__(
        self,
        model: VGSSM,
        graph,
        learning_rate=1e-3,
        weight_decay=1e-4,
        beta_ce=0.01,
        beta_z=0.001,
        horizon_weighting='linear',
        warmup_epochs=5,
        max_epochs=50,
        norm_stats=None,
        loss_type='mse',
        huber_delta=1.0,
        free_bits_ce=0.1,
        free_bits_z=0.05,
        # Physics loss settings
        use_physics_loss=False,
        physics_local_weight=0.1,
        physics_global_weight=0.01,
        dt_seconds=300.0,
        physics_loss_mode='residual',  # residual | light | legacy
        physics_residual_huber_delta=0.1,
        # Soft boundary loss settings
        use_boundary_loss=True,  # Enable soft boundary loss by default
        boundary_loss_weight=0.1,  # Weight for boundary penalty
        boundary_delta=10.0,  # Huber delta for boundary loss
        output_bounds_1d=None,  # (min, max) for 1D nodes
        output_bounds_2d=None,  # (min, max) for 2D nodes
        # Curriculum settings
        use_curriculum=False,
        curriculum_scheduler=None,
        # Future inlet availability control for train-test alignment
        future_inlet_mode_train='missing',  # full | missing | mixed
        future_inlet_keep_prob=0.0,  # used when mode=mixed
        # Reconstruction balancing
        recon_balance_mode='equal',  # equal | sum
        recon_weight_1d=0.5,
        recon_weight_2d=0.5,
        horizon_weight_by_valid_count=False,
    ):
        super().__init__()
        self.model = model
        self.graph = graph
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta_ce = beta_ce
        self.beta_z = beta_z
        self.horizon_weighting = horizon_weighting
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.norm_stats = norm_stats
        self.loss_type = loss_type
        self.huber_delta = huber_delta
        self.free_bits_ce = free_bits_ce
        self.free_bits_z = free_bits_z

        # Physics settings with automatic graph-size scaling
        self.use_physics_loss = use_physics_loss
        self.physics_loss_mode = str(physics_loss_mode)
        if self.physics_loss_mode not in {'residual', 'light', 'legacy'}:
            print(f"Warning: unknown physics_loss_mode={self.physics_loss_mode}, falling back to residual")
            self.physics_loss_mode = 'residual'
        self.physics_residual_huber_delta = float(max(physics_residual_huber_delta, 1e-4))

        # Scale physics loss weights by inverse sqrt of graph size for stability
        if use_physics_loss and graph is not None:
            num_1d = graph['1d'].x.shape[0] if hasattr(graph, '1d') else 17
            num_2d = graph['2d'].x.shape[0] if hasattr(graph, '2d') else 3716
            total_nodes = num_1d + num_2d
            # Reference: Model 1 has ~3733 nodes, Model 2 has ~35000 nodes
            reference_nodes = 3733.0
            scale_factor = math.sqrt(reference_nodes / max(total_nodes, 1))
            self.physics_local_weight = physics_local_weight * scale_factor
            self.physics_global_weight = physics_global_weight * scale_factor
            print(f"Physics loss scaled by {scale_factor:.4f} for {total_nodes} nodes")
        else:
            self.physics_local_weight = physics_local_weight
            self.physics_global_weight = physics_global_weight

        if use_physics_loss:
            # Keep all variants available; select by physics_loss_mode.
            self.light_physics = LightPhysicsRegularizer()
            self.spatial_smoothness = SpatialSmoothness(percentile=95.0)
            self.physics_residual = PhysicsResidualLoss(
                dt_seconds=dt_seconds,
                huber_delta=self.physics_residual_huber_delta,
            )
            # Learnable source scaling to align dataset units with flux units.
            self.physics_source_log_scale_1d = nn.Parameter(torch.tensor(0.0))
            self.physics_source_log_scale_2d = nn.Parameter(torch.tensor(-2.0))
            # Keep conductances near physically sensible priors unless data says otherwise.
            self.physics_conductance_prior_weight = 1e-4

            # Legacy light-physics coefficients (used only in light mode).
            self.physics_temporal_weight = 0.0001  # 1000x smaller than before
            self.physics_variance_weight = 0.001   # Encourage variance preservation
            self.physics_spatial_weight = 0.00001  # Very light spatial smoothing

            # Keep old modules for backward compatibility
            self.simplified_physics_loss = SimplifiedPhysicsLoss(dt_seconds)
            self.local_mass_loss = LocalMassConservationLoss(dt_seconds)
            self.global_mass_loss = GlobalMassConservationLoss(dt_seconds)
            self.use_light_physics = self.physics_loss_mode == 'light'
            self.use_simplified_physics = False

        # Soft boundary loss (replaces hard tanh/sigmoid bounds)
        self.use_boundary_loss = use_boundary_loss
        self.boundary_loss_weight = boundary_loss_weight
        self.boundary_delta = boundary_delta
        if use_boundary_loss and output_bounds_1d is not None:
            self.boundary_loss_1d = SoftBoundaryLoss(
                min_val=output_bounds_1d[0],
                max_val=output_bounds_1d[1],
                weight=boundary_loss_weight,
                delta=boundary_delta
            )
        else:
            self.boundary_loss_1d = None

        if use_boundary_loss and output_bounds_2d is not None:
            self.boundary_loss_2d = SoftBoundaryLoss(
                min_val=output_bounds_2d[0],
                max_val=output_bounds_2d[1],
                weight=boundary_loss_weight,
                delta=boundary_delta
            )
        else:
            self.boundary_loss_2d = None

        # Curriculum settings
        self.use_curriculum = use_curriculum
        self.curriculum_scheduler = curriculum_scheduler
        self.future_inlet_mode_train = future_inlet_mode_train
        self.future_inlet_keep_prob = future_inlet_keep_prob
        self.recon_balance_mode = recon_balance_mode
        self.horizon_weight_by_valid_count = horizon_weight_by_valid_count

        rw_1d = max(float(recon_weight_1d), 0.0)
        rw_2d = max(float(recon_weight_2d), 0.0)
        if rw_1d + rw_2d <= 0:
            rw_1d, rw_2d = 0.5, 0.5
        norm = rw_1d + rw_2d
        self.recon_weight_1d = rw_1d / norm
        self.recon_weight_2d = rw_2d / norm

        # Extract node areas from graph for physics loss
        self._setup_node_areas()

        self.save_hyperparameters(ignore=['model', 'graph', 'norm_stats', 'curriculum_scheduler'])

    @staticmethod
    def _sanitize_metric(x: torch.Tensor) -> torch.Tensor:
        if torch.isnan(x) or torch.isinf(x):
            return torch.tensor(1e6, device=x.device)
        return x

    def _compute_std_rmse(self, pred_1d, pred_2d, target_1d, target_2d, target_mask=None):
        horizon = min(pred_1d.shape[1], target_1d.shape[1])
        pred_1d = pred_1d[:, :horizon]
        pred_2d = pred_2d[:, :horizon]
        target_1d = target_1d[:, :horizon]
        target_2d = target_2d[:, :horizon]

        if target_mask is not None:
            target_mask = target_mask[:, :horizon].to(device=pred_1d.device, dtype=pred_1d.dtype)
            mask_1d = target_mask.unsqueeze(-1).expand_as(pred_1d)
            mask_2d = target_mask.unsqueeze(-1).expand_as(pred_2d)

            valid_1d = mask_1d.sum()
            valid_2d = mask_2d.sum()
            if valid_1d <= 0 or valid_2d <= 0:
                bad = torch.tensor(1e6, device=pred_1d.device, dtype=pred_1d.dtype)
                return bad, bad, bad, pred_1d, pred_2d

            mse_1d = ((pred_1d - target_1d) ** 2 * mask_1d).sum() / valid_1d
            mse_2d = ((pred_2d - target_2d) ** 2 * mask_2d).sum() / valid_2d
            rmse_1d = torch.sqrt(mse_1d)
            rmse_2d = torch.sqrt(mse_2d)
        else:
            rmse_1d = torch.sqrt(F.mse_loss(pred_1d, target_1d))
            rmse_2d = torch.sqrt(F.mse_loss(pred_2d, target_2d))
        # Targets are already normalized by target std in FloodEventDataset.
        # Therefore RMSE in this space is already standardized RMSE.
        std_rmse = (rmse_1d + rmse_2d) / 2
        std_rmse = self._sanitize_metric(std_rmse)

        return rmse_1d, rmse_2d, std_rmse, pred_1d, pred_2d

    def _setup_node_areas(self):
        """Extract node areas from graph for mass conservation computation.
        Note: Areas are computed lazily in _get_node_areas to handle device/shape issues.
        """
        # Store graph reference for lazy computation
        self._graph_ref = self.graph

    def _get_node_areas(self, num_1d: int, num_2d: int, device):
        """Lazily compute node areas with correct shapes and device."""
        try:
            if hasattr(self._graph_ref, '1d') and hasattr(self._graph_ref['1d'], 'x'):
                x_1d = self._graph_ref['1d'].x
                if x_1d.shape[0] == num_1d and x_1d.shape[1] > 5:
                    areas_1d = x_1d[:, 5].clamp(min=1.0).to(device)
                else:
                    areas_1d = torch.ones(num_1d, device=device)
            else:
                areas_1d = torch.ones(num_1d, device=device)
        except Exception:
            areas_1d = torch.ones(num_1d, device=device)

        try:
            if hasattr(self._graph_ref, '2d') and hasattr(self._graph_ref['2d'], 'x'):
                x_2d = self._graph_ref['2d'].x
                if x_2d.shape[0] == num_2d and x_2d.shape[1] > 2:
                    areas_2d = x_2d[:, 2].clamp(min=1.0).to(device)
                else:
                    areas_2d = torch.ones(num_2d, device=device)
            else:
                areas_2d = torch.ones(num_2d, device=device)
        except Exception:
            areas_2d = torch.ones(num_2d, device=device)

        return areas_1d, areas_2d

    def _denormalize_target(self, x: torch.Tensor, target_key: str) -> torch.Tensor:
        """Convert normalized target-space tensor back to physical units."""
        if self.norm_stats is None:
            return x
        stats = self.norm_stats.get(target_key, None)
        if stats is None:
            return x
        mean = float(stats.get('mean', 0.0))
        std = max(float(stats.get('std', 1.0)), 1e-8)
        return x * std + mean

    def _denormalize_dynamic_channel(self, x: Optional[torch.Tensor], domain_key: str, channel_idx: int) -> Optional[torch.Tensor]:
        """Convert normalized dynamic input channel back to physical units."""
        if x is None or self.norm_stats is None:
            return x
        stats = self.norm_stats.get(domain_key, None)
        if stats is None:
            return x
        mean_arr = np.asarray(stats.get('mean'))
        std_arr = np.asarray(stats.get('std'))
        if mean_arr.ndim == 0 or std_arr.ndim == 0:
            return x
        if channel_idx < 0 or channel_idx >= mean_arr.shape[0] or channel_idx >= std_arr.shape[0]:
            return x
        mean = float(mean_arr[channel_idx])
        std = max(float(std_arr[channel_idx]), 1e-8)
        return x * std + mean

    def forward(self, input_1d, input_2d, **kwargs):
        return self.model(self.graph, input_1d, input_2d, **kwargs)

    def _compute_loss(self, pred, target, mask=None):
        if mask is None:
            if self.loss_type == 'huber':
                return F.huber_loss(pred, target, delta=self.huber_delta)
            return F.mse_loss(pred, target)

        # mask can be [batch], [batch, num_nodes], or already element-wise.
        # Expand to element-wise mask so normalization matches actual prediction
        # element count instead of only batch count.
        mask = mask.to(dtype=pred.dtype, device=pred.device)
        while mask.dim() < pred.dim():
            mask = mask.unsqueeze(-1)
        if mask.shape != pred.shape:
            mask = mask.expand_as(pred)
        valid = mask.sum()
        if valid <= 0:
            return torch.tensor(0.0, device=pred.device, dtype=pred.dtype)

        if self.loss_type == 'huber':
            loss_elem = F.huber_loss(pred, target, delta=self.huber_delta, reduction='none')
            return (loss_elem * mask).sum() / valid

        sq = (pred - target) ** 2
        return (sq * mask).sum() / valid

    def _get_horizon_weights(self, horizon, device):
        if self.horizon_weighting == 'uniform':
            return torch.ones(horizon, device=device) / horizon
        elif self.horizon_weighting == 'linear':
            weights = torch.arange(1, horizon + 1, device=device, dtype=torch.float)
            return weights / weights.sum()
        elif self.horizon_weighting == 'exp':
            weights = torch.exp(torch.arange(horizon, device=device, dtype=torch.float) * 0.05)
            return weights / weights.sum()
        return torch.ones(horizon, device=device) / horizon

    def _kl_z0(self, mu, logvar, free_bits=0.0):
        kl_per_dim = 0.5 * (-1 - logvar + mu.pow(2) + logvar.exp())
        if free_bits > 0:
            kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
        return kl_per_dim.sum(dim=-1).mean()

    def _kl_divergence(self, mean_q, logvar_q, mean_p, logvar_p, free_bits=0.0):
        var_q = torch.exp(logvar_q)
        var_p = torch.exp(logvar_p)
        kl_per_dim = 0.5 * (logvar_p - logvar_q + var_q / var_p + (mean_q - mean_p) ** 2 / var_p - 1)
        if free_bits > 0:
            kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)
        return kl_per_dim.sum(dim=-1).mean()

    def _compute_physics_loss(self, outputs, target_1d, target_2d, future_rainfall, future_inlet_flow, target_mask=None):
        """Compute physics-informed losses.

        Modes:
        - residual: conservation residual in physical units (recommended).
        - light: ultra-light temporal/variance regularization.
        - legacy: previous flow-head-based formulation.
        """
        device = target_1d.device

        if not self.use_physics_loss:
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

        pred_1d_norm = outputs['pred_1d'][..., 0]  # [batch, horizon, num_1d]
        pred_2d_norm = outputs['pred_2d'][..., 0]  # [batch, horizon, num_2d]

        # Check for NaN early
        if torch.isnan(pred_1d_norm).any() or torch.isnan(pred_2d_norm).any():
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

        num_1d = pred_1d_norm.shape[2]
        num_2d = pred_2d_norm.shape[2]

        # Get node areas
        node_areas_1d, node_areas_2d = self._get_node_areas(num_1d, num_2d, device)

        # Get 2D surface edges for spatial smoothness
        edge_index_2d = None
        if 'edge_index_dict' in outputs:
            for edge_type, edge_index in outputs['edge_index_dict'].items():
                if edge_type[1] == 'surface':  # ('2d', 'surface', '2d')
                    edge_index_2d = edge_index
                    break

        # Light mode: very soft regularization.
        if self.physics_loss_mode == 'light':
            temporal_reg, variance_reg = self.light_physics(
                pred_1d=pred_1d_norm,
                pred_2d=pred_2d_norm,
                target_1d=target_1d.squeeze(-1) if target_1d.dim() == 4 else target_1d,
                target_2d=target_2d.squeeze(-1) if target_2d.dim() == 4 else target_2d,
                edge_index_2d=edge_index_2d,
            )
            if edge_index_2d is not None and hasattr(self, 'spatial_smoothness'):
                spatial_reg = self.spatial_smoothness(pred_2d_norm, edge_index_2d)
            else:
                spatial_reg = torch.tensor(0.0, device=device)

            physics_loss_1 = self.physics_temporal_weight * temporal_reg
            physics_loss_2 = self.physics_variance_weight * variance_reg + self.physics_spatial_weight * spatial_reg
            max_loss = 1.0
            physics_loss_1 = physics_loss_1.clamp(0, max_loss) if not torch.isnan(physics_loss_1) else torch.tensor(0.0, device=device)
            physics_loss_2 = physics_loss_2.clamp(0, max_loss) if not torch.isnan(physics_loss_2) else torch.tensor(0.0, device=device)
            return physics_loss_1, physics_loss_2

        # Residual mode: compute conservation in physical units.
        if self.physics_loss_mode == 'residual':
            edge_index_dict = outputs.get('edge_index_dict', None)
            if edge_index_dict is None:
                return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

            pred_1d = self._denormalize_target(pred_1d_norm, 'target_1d')
            pred_2d = self._denormalize_target(pred_2d_norm, 'target_2d')
            rainfall_phys = self._denormalize_dynamic_channel(future_rainfall, '2d', 0)
            inlet_phys = self._denormalize_dynamic_channel(future_inlet_flow, '1d', 1)

            horizon = min(pred_1d.shape[1], pred_2d.shape[1])
            if target_mask is not None:
                target_mask = target_mask[:, :horizon].to(device=device, dtype=pred_1d.dtype)

            physics_subsample_rate = int(max(getattr(self.model, 'physics_subsample_rate', 5), 1))
            source_scale_1d = torch.exp(self.physics_source_log_scale_1d).to(device=device)
            source_scale_2d = torch.exp(self.physics_source_log_scale_2d).to(device=device)
            node_areas_all = torch.cat([node_areas_1d, node_areas_2d], dim=0)

            local_losses = []
            global_losses = []
            global_abs_residuals = []
            batch_size = pred_1d.shape[0]

            for t in range(1, horizon):
                if (t % physics_subsample_rate != 0) and (t != horizon - 1):
                    continue
                if target_mask is not None and target_mask[:, t].sum() <= 0:
                    continue

                h_curr_1d = pred_1d[:, t].clamp(-1e4, 1e4)
                h_prev_1d = pred_1d[:, t - 1].clamp(-1e4, 1e4)
                h_curr_2d = pred_2d[:, t].clamp(-1e4, 1e4)
                h_prev_2d = pred_2d[:, t - 1].clamp(-1e4, 1e4)

                flows_t = self.model.compute_physics_flows(h_curr_1d, h_curr_2d, edge_index_dict)
                if not flows_t:
                    continue

                in_1d, out_1d, in_2d, out_2d = self.model.aggregate_flows_to_nodes(
                    flows_t, edge_index_dict, num_1d, num_2d, batch_size, device
                )

                if rainfall_phys is not None and t < rainfall_phys.shape[1]:
                    rain_t = rainfall_phys[:, t]
                    if rain_t.dim() == 3 and rain_t.shape[-1] == 1:
                        rain_t = rain_t[..., 0]
                    rain_t = rain_t.clamp(min=0.0)
                else:
                    rain_t = torch.zeros(batch_size, num_2d, device=device, dtype=pred_2d.dtype)

                if inlet_phys is not None and t < inlet_phys.shape[1]:
                    inlet_t = inlet_phys[:, t]
                    if inlet_t.dim() == 3 and inlet_t.shape[-1] == 1:
                        inlet_t = inlet_t[..., 0]
                else:
                    inlet_t = torch.zeros(batch_size, num_1d, device=device, dtype=pred_1d.dtype)

                source_1d = source_scale_1d * inlet_t
                source_2d = source_scale_2d * rain_t * node_areas_2d.unsqueeze(0)

                h_curr_all = torch.cat([h_curr_1d, h_curr_2d], dim=1)
                h_prev_all = torch.cat([h_prev_1d, h_prev_2d], dim=1)
                in_all = torch.cat([in_1d, in_2d], dim=1)
                out_all = torch.cat([out_1d, out_2d], dim=1)
                source_all = torch.cat([source_1d, source_2d], dim=1)

                local_t, global_t, global_abs_t = self.physics_residual(
                    h_curr=h_curr_all,
                    h_prev=h_prev_all,
                    node_areas=node_areas_all,
                    flows_in=in_all,
                    flows_out=out_all,
                    sources=source_all,
                )

                if not torch.isnan(local_t):
                    local_losses.append(local_t)
                if not torch.isnan(global_t):
                    global_losses.append(global_t)
                if not torch.isnan(global_abs_t):
                    global_abs_residuals.append(global_abs_t)

            if not local_losses or not global_losses:
                return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

            local_loss = torch.stack(local_losses).mean()
            global_loss = torch.stack(global_losses).mean()
            if global_abs_residuals:
                self.log('train/physics_global_abs_residual', torch.stack(global_abs_residuals).mean())

            # Regularize conductance to avoid degenerate flow scaling.
            if hasattr(self.model, 'physics_flow'):
                conductance_prior = (
                    (self.model.physics_flow.log_K_pipe - 0.0) ** 2
                    + (self.model.physics_flow.log_K_surface + 1.0) ** 2
                    + (self.model.physics_flow.log_K_coupling + 0.5) ** 2
                )
                local_loss = local_loss + self.physics_conductance_prior_weight * conductance_prior

            return local_loss, global_loss

        # Legacy mode: flow-based conservation (kept for backward compatibility).
        if 'flows' not in outputs:
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

        batch = target_1d.shape[0]
        flows_dict = outputs['flows']
        edge_index_dict = outputs['edge_index_dict']

        available_timesteps = sorted(flows_dict.keys())
        if len(available_timesteps) < 2:
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

        local_losses = []
        global_losses = []
        horizon = pred_1d_norm.shape[1]
        max_loss = 1000.0

        for i in range(1, len(available_timesteps)):
            t_curr = available_timesteps[i]
            t_prev = available_timesteps[i - 1]
            if t_curr >= horizon or t_prev >= horizon:
                continue

            flows_t = flows_dict[t_curr]
            has_nan_flow = any(torch.isnan(f).any() for f in flows_t.values())
            if has_nan_flow:
                continue

            in_1d, out_1d, in_2d, out_2d = self.model.aggregate_flows_to_nodes(
                flows_t, edge_index_dict, num_1d, num_2d, batch, device
            )
            in_1d = in_1d.clamp(-max_loss, max_loss)
            out_1d = out_1d.clamp(-max_loss, max_loss)
            in_2d = in_2d.clamp(-max_loss, max_loss)
            out_2d = out_2d.clamp(-max_loss, max_loss)

            h_curr_1d = pred_1d_norm[:, t_curr].clamp(-1e4, 1e4)
            h_prev_1d = pred_1d_norm[:, t_prev].clamp(-1e4, 1e4)
            h_curr_2d = pred_2d_norm[:, t_curr].clamp(-1e4, 1e4)
            h_prev_2d = pred_2d_norm[:, t_prev].clamp(-1e4, 1e4)

            source_2d = future_rainfall[:, t_curr, :, 0].clamp(0, 1e3) if future_rainfall is not None and t_curr < future_rainfall.shape[1] else torch.zeros(batch, num_2d, device=device)
            source_1d = future_inlet_flow[:, t_curr, :, 0].clamp(0, 1e3) if future_inlet_flow is not None and t_curr < future_inlet_flow.shape[1] else torch.zeros(batch, num_1d, device=device)

            local_loss_1d = self.local_mass_loss(h_curr_1d, h_prev_1d, node_areas_1d, in_1d, out_1d, source_1d)
            local_loss_2d = self.local_mass_loss(h_curr_2d, h_prev_2d, node_areas_2d, in_2d, out_2d, source_2d)
            local_loss_combined = (local_loss_1d + local_loss_2d) / 2
            if not torch.isnan(local_loss_combined) and local_loss_combined < max_loss:
                local_losses.append(local_loss_combined.clamp(0, max_loss))

            boundary_inflow = source_1d.sum(dim=1) + source_2d.sum(dim=1)
            boundary_outflow = torch.zeros(batch, device=device)
            total_source = source_2d.sum(dim=1)

            h_curr_all = torch.cat([h_curr_1d, h_curr_2d], dim=1)
            h_prev_all = torch.cat([h_prev_1d, h_prev_2d], dim=1)
            node_areas_all = torch.cat([node_areas_1d, node_areas_2d])
            global_loss_t = self.global_mass_loss(h_curr_all, h_prev_all, node_areas_all, boundary_inflow, boundary_outflow, total_source)
            if not torch.isnan(global_loss_t) and global_loss_t < max_loss:
                global_losses.append(global_loss_t.clamp(0, max_loss))

        local_loss = torch.stack(local_losses).mean().clamp(0, max_loss) if local_losses else torch.tensor(0.0, device=device)
        global_loss = torch.stack(global_losses).mean().clamp(0, max_loss) if global_losses else torch.tensor(0.0, device=device)
        if torch.isnan(local_loss):
            local_loss = torch.tensor(0.0, device=device)
        if torch.isnan(global_loss):
            global_loss = torch.tensor(0.0, device=device)

        return local_loss, global_loss

    def training_step(self, batch, batch_idx):
        input_1d = batch['input_1d']
        input_2d = batch['input_2d']
        target_1d = batch['target_1d']
        target_2d = batch['target_2d']
        target_mask = batch.get('target_mask')
        future_rainfall = batch.get('future_rainfall')
        future_inlet_flow = batch.get('future_inlet_flow')
        future_inlet_flow_for_model = future_inlet_flow
        if self.future_inlet_mode_train == 'missing':
            future_inlet_flow_for_model = None
        elif self.future_inlet_mode_train == 'mixed':
            if torch.rand(1, device=input_1d.device).item() > float(self.future_inlet_keep_prob):
                future_inlet_flow_for_model = None

        prefix_len = batch.get('prefix_len', 10)
        if isinstance(prefix_len, torch.Tensor):
            prefix_len = prefix_len[0].item()

        # Get rollout length from curriculum if enabled
        if self.use_curriculum and self.curriculum_scheduler is not None:
            rollout_len = self.curriculum_scheduler.current_rollout_len
        else:
            rollout_len = None
        batch_rollout_len = batch.get('rollout_len')
        if isinstance(batch_rollout_len, torch.Tensor):
            batch_rollout_len = int(batch_rollout_len.max().item())
        elif batch_rollout_len is not None:
            batch_rollout_len = int(batch_rollout_len)
        if batch_rollout_len is not None:
            rollout_len = batch_rollout_len if rollout_len is None else min(int(rollout_len), int(batch_rollout_len))

        # Forward pass
        # Only the legacy physics mode requires explicit EdgeFlowHead flow tensors.
        use_legacy_flows = self.use_physics_loss and self.physics_loss_mode == 'legacy'
        outputs = self.model(
            self.graph, input_1d, input_2d,
            prefix_len=prefix_len,
            future_rainfall=future_rainfall,
            future_inlet_flow=future_inlet_flow_for_model,
            rollout_len=rollout_len,
            return_flows=use_legacy_flows,  # Only compute EdgeFlowHead flows for legacy mode
            deterministic_latent=False,
        )

        pred_1d = outputs['pred_1d'][..., 0]
        pred_2d = outputs['pred_2d'][..., 0]

        if target_1d.dim() == 4:
            target_1d = target_1d.squeeze(-1)
        if target_2d.dim() == 4:
            target_2d = target_2d.squeeze(-1)
        if target_mask is None:
            target_mask = torch.ones(target_1d.shape[:2], device=target_1d.device, dtype=target_1d.dtype)
        else:
            target_mask = target_mask.to(device=target_1d.device, dtype=target_1d.dtype)

        # Adjust horizon for curriculum
        if rollout_len is not None:
            horizon = min(rollout_len, pred_1d.shape[1], target_1d.shape[1])
        else:
            horizon = min(pred_1d.shape[1], target_1d.shape[1])

        pred_1d = pred_1d[:, :horizon]
        pred_2d = pred_2d[:, :horizon]
        target_1d = target_1d[:, :horizon]
        target_2d = target_2d[:, :horizon]
        target_mask = target_mask[:, :horizon]
        if rollout_len is not None and rollout_len < horizon:
            target_mask[:, rollout_len:] = 0.0

        horizon_weights = self._get_horizon_weights(horizon, pred_1d.device)
        valid_counts = target_mask.sum(dim=0).to(dtype=horizon_weights.dtype, device=horizon_weights.device)
        if self.horizon_weight_by_valid_count:
            horizon_weights = horizon_weights * valid_counts
        else:
            valid_steps = (valid_counts > 0).to(dtype=horizon_weights.dtype, device=horizon_weights.device)
            horizon_weights = horizon_weights * valid_steps
        if horizon_weights.sum() > 0:
            horizon_weights = horizon_weights / horizon_weights.sum()

        losses_1d = []
        losses_2d = []
        for h in range(horizon):
            if target_mask[:, h].sum() <= 0:
                continue
            loss_1d_h = self._compute_loss(pred_1d[:, h], target_1d[:, h], mask=target_mask[:, h])
            loss_2d_h = self._compute_loss(pred_2d[:, h], target_2d[:, h], mask=target_mask[:, h])
            losses_1d.append(loss_1d_h * horizon_weights[h])
            losses_2d.append(loss_2d_h * horizon_weights[h])

        if losses_1d:
            loss_recon_1d = torch.stack(losses_1d).sum()
            loss_recon_2d = torch.stack(losses_2d).sum()
        else:
            loss_recon_1d = torch.tensor(0.0, device=pred_1d.device, dtype=pred_1d.dtype)
            loss_recon_2d = torch.tensor(0.0, device=pred_1d.device, dtype=pred_1d.dtype)
        if self.recon_balance_mode == 'sum':
            loss_recon = loss_recon_1d + loss_recon_2d
        else:
            loss_recon = self.recon_weight_1d * loss_recon_1d + self.recon_weight_2d * loss_recon_2d

        kl_anneal = min(1.0, self.current_epoch / max(1, self.warmup_epochs))

        if self.model.use_event_latent:
            kl_ce = self._kl_divergence(
                outputs['c_e_mean'], outputs['c_e_logvar'],
                self.model.event_prior_mean, self.model.event_prior_logvar,
                free_bits=self.free_bits_ce,
            )
            kl_ce_weight = kl_anneal * self.beta_ce
        else:
            kl_ce = torch.tensor(0.0, device=pred_1d.device)
            kl_ce_weight = 0.0

        kl_z0_1d = self._kl_z0(outputs['z0_mu_1d'], outputs['z0_logvar_1d'], free_bits=self.free_bits_z)
        kl_z0_2d = self._kl_z0(outputs['z0_mu_2d'], outputs['z0_logvar_2d'], free_bits=self.free_bits_z)
        kl_z0 = kl_z0_1d + kl_z0_2d
        kl_z0_weight = kl_anneal * self.beta_z

        total_loss = loss_recon + kl_ce_weight * kl_ce + kl_z0_weight * kl_z0

        # Physics losses with adaptive weighting
        if self.use_physics_loss:
            local_loss, global_loss = self._compute_physics_loss(
                outputs, target_1d, target_2d, future_rainfall, future_inlet_flow_for_model, target_mask=target_mask
            )

            # Skip physics loss in early epochs to let model learn basic task first
            physics_warmup_epochs = 3
            if self.current_epoch < physics_warmup_epochs:
                physics_anneal = 0.0
            else:
                # Gradually increase physics loss weight after warmup
                physics_anneal = min(1.0, (self.current_epoch - physics_warmup_epochs) / max(1, self.warmup_epochs))

            # Adaptive scaling: normalize physics loss relative to reconstruction loss
            # This prevents physics loss from dominating or being negligible
            physics_loss_raw = local_loss + global_loss
            if self.physics_loss_mode == 'residual':
                # Residual mode is already scale-normalized by construction.
                adaptive_physics_weight = torch.tensor(1.0, device=pred_1d.device, dtype=pred_1d.dtype)
            elif physics_loss_raw > 0 and loss_recon > 0 and not torch.isnan(physics_loss_raw):
                loss_ratio = (loss_recon.detach() / (physics_loss_raw.detach() + 1e-8)).clamp(0.01, 100.0)
                adaptive_physics_weight = loss_ratio * 0.1
            else:
                adaptive_physics_weight = torch.tensor(1.0, device=pred_1d.device, dtype=pred_1d.dtype)

            # Apply scaled physics loss
            physics_contribution = physics_anneal * adaptive_physics_weight * (
                self.physics_local_weight * local_loss +
                self.physics_global_weight * global_loss
            )

            # Only add physics loss if it's valid (not NaN or too large)
            if not torch.isnan(physics_contribution) and physics_contribution < 100.0:
                total_loss = total_loss + physics_contribution

            # For simplified physics (v6+): local_loss=volume_balance, global_loss=smoothness
            # For legacy: local_loss=local_conservation, global_loss=global_conservation
            self.log('train/physics_volume', local_loss)  # Volume balance loss
            self.log('train/physics_smooth', global_loss)  # Smoothness loss
            self.log('train/physics_contribution', physics_contribution)
            self.log('train/adaptive_physics_weight', adaptive_physics_weight)
            self.log('train/physics_mode_residual', float(self.physics_loss_mode == 'residual'))
            self.log('train/physics_mode_light', float(self.physics_loss_mode == 'light'))
            self.log('train/physics_mode_legacy', float(self.physics_loss_mode == 'legacy'))
            if hasattr(self, 'physics_source_log_scale_1d'):
                self.log('train/physics_source_scale_1d', torch.exp(self.physics_source_log_scale_1d))
            if hasattr(self, 'physics_source_log_scale_2d'):
                self.log('train/physics_source_scale_2d', torch.exp(self.physics_source_log_scale_2d))
            if hasattr(self.model, 'physics_flow'):
                self.log('train/physics_K_pipe', torch.exp(self.model.physics_flow.log_K_pipe))
                self.log('train/physics_K_surface', torch.exp(self.model.physics_flow.log_K_surface))
                self.log('train/physics_K_coupling', torch.exp(self.model.physics_flow.log_K_coupling))

        # Soft boundary loss - penalizes predictions outside physical bounds
        if self.use_boundary_loss:
            boundary_loss = torch.tensor(0.0, device=pred_1d.device)
            valid_mask_nodes_1d = target_mask.unsqueeze(-1).expand_as(pred_1d) > 0
            valid_mask_nodes_2d = target_mask.unsqueeze(-1).expand_as(pred_2d) > 0
            if self.boundary_loss_1d is not None:
                if valid_mask_nodes_1d.any():
                    boundary_loss_1d = self.boundary_loss_1d(pred_1d[valid_mask_nodes_1d])
                else:
                    boundary_loss_1d = torch.tensor(0.0, device=pred_1d.device, dtype=pred_1d.dtype)
                boundary_loss = boundary_loss + boundary_loss_1d
                self.log('train/boundary_loss_1d', boundary_loss_1d)
            if self.boundary_loss_2d is not None:
                if valid_mask_nodes_2d.any():
                    boundary_loss_2d = self.boundary_loss_2d(pred_2d[valid_mask_nodes_2d])
                else:
                    boundary_loss_2d = torch.tensor(0.0, device=pred_2d.device, dtype=pred_2d.dtype)
                boundary_loss = boundary_loss + boundary_loss_2d
                self.log('train/boundary_loss_2d', boundary_loss_2d)

            if boundary_loss > 0:
                total_loss = total_loss + boundary_loss
                self.log('train/boundary_loss', boundary_loss)

        self.log('train/loss', total_loss, prog_bar=True)
        self.log('train/loss_recon', loss_recon)
        self.log('train/loss_recon_1d', loss_recon_1d)
        self.log('train/loss_recon_2d', loss_recon_2d)
        self.log('train/kl_ce', kl_ce)
        self.log('train/kl_z0', kl_z0)
        self.log('train/future_inlet_used', float(future_inlet_flow_for_model is not None))
        self.log('train/valid_target_frac', target_mask.mean())
        self.log('train/recon_weight_1d', float(self.recon_weight_1d))
        self.log('train/recon_weight_2d', float(self.recon_weight_2d))
        self.log('train/recon_balance_sum_mode', float(self.recon_balance_mode == 'sum'))
        self.log('train/horizon_weight_by_valid_count', float(self.horizon_weight_by_valid_count))

        if self.use_curriculum and self.curriculum_scheduler is not None:
            self.log('train/rollout_len', float(self.curriculum_scheduler.current_rollout_len))

        return total_loss

    def validation_step(self, batch, batch_idx):
        input_1d = batch['input_1d']
        input_2d = batch['input_2d']
        target_1d = batch['target_1d']
        target_2d = batch['target_2d']
        target_mask = batch.get('target_mask')
        future_rainfall = batch.get('future_rainfall')

        prefix_len = batch.get('prefix_len', 10)
        if isinstance(prefix_len, torch.Tensor):
            prefix_len = prefix_len[0].item()

        if self.use_curriculum and self.curriculum_scheduler is not None:
            rollout_len = int(self.curriculum_scheduler.current_rollout_len)
        else:
            rollout_len = None
        batch_rollout_len = batch.get('rollout_len')
        if isinstance(batch_rollout_len, torch.Tensor):
            batch_rollout_len = int(batch_rollout_len.max().item())
        elif batch_rollout_len is not None:
            batch_rollout_len = int(batch_rollout_len)
        if batch_rollout_len is not None:
            rollout_len = batch_rollout_len if rollout_len is None else min(int(rollout_len), int(batch_rollout_len))

        if target_1d.dim() == 4:
            target_1d = target_1d.squeeze(-1)
        if target_2d.dim() == 4:
            target_2d = target_2d.squeeze(-1)
        if target_mask is None:
            target_mask = torch.ones(target_1d.shape[:2], device=target_1d.device, dtype=target_1d.dtype)
        else:
            target_mask = target_mask.to(device=target_1d.device, dtype=target_1d.dtype)

        full_horizon = min(int(self.model.prediction_horizon), int(target_1d.shape[1]))
        if self.use_curriculum and self.curriculum_scheduler is not None:
            # Always evaluate full horizon during curriculum.
            eval_rollout_len = full_horizon
        else:
            # For variable-horizon batches, only unroll as far as needed.
            eval_rollout_len = rollout_len
        with torch.no_grad():
            outputs_eval = self.model(
                self.graph,
                input_1d,
                input_2d,
                prefix_len=prefix_len,
                future_rainfall=future_rainfall,
                rollout_len=eval_rollout_len,
                deterministic_latent=True,
            )

        pred_1d_full = outputs_eval['pred_1d'][..., 0]
        pred_2d_full = outputs_eval['pred_2d'][..., 0]
        full_horizon = min(full_horizon, int(pred_1d_full.shape[1]))
        target_mask_full = target_mask[:, :full_horizon]
        target_mask_curr = target_mask_full
        if rollout_len is not None and rollout_len < full_horizon:
            target_mask_curr = target_mask_full.clone()
            target_mask_curr[:, rollout_len:] = 0.0

        rmse_1d_full, rmse_2d_full, std_rmse_full, pred_1d_eval, pred_2d_eval = self._compute_std_rmse(
            pred_1d_full, pred_2d_full, target_1d, target_2d, target_mask=target_mask_full
        )

        rmse_1d_monitor = rmse_1d_full
        rmse_2d_monitor = rmse_2d_full
        std_rmse_monitor = std_rmse_full

        if rollout_len is not None and rollout_len < full_horizon:
            rmse_1d_curr, rmse_2d_curr, std_rmse_curr, _, _ = self._compute_std_rmse(
                pred_1d_full[:, :rollout_len],
                pred_2d_full[:, :rollout_len],
                target_1d[:, :rollout_len],
                target_2d[:, :rollout_len],
                target_mask=target_mask_curr[:, :rollout_len],
            )
        else:
            rmse_1d_curr = rmse_1d_full
            rmse_2d_curr = rmse_2d_full
            std_rmse_curr = std_rmse_full

        self.log('val/std_rmse_full', std_rmse_full)
        self.log('val/std_rmse_curr', std_rmse_curr)
        if rollout_len is not None:
            self.log('val/rollout_len_curr', float(rollout_len))
        self.log('val/rollout_len_full', float(full_horizon))
        self.log('val/valid_target_frac', target_mask_full.mean())

        # Track boundary violations during validation
        if self.use_boundary_loss:
            with torch.no_grad():
                valid_mask_nodes_1d = target_mask_full.unsqueeze(-1).expand_as(pred_1d_eval) > 0
                valid_mask_nodes_2d = target_mask_full.unsqueeze(-1).expand_as(pred_2d_eval) > 0
                if self.boundary_loss_1d is not None:
                    if valid_mask_nodes_1d.any():
                        pred_1d_valid = pred_1d_eval[valid_mask_nodes_1d]
                        boundary_loss_1d = self.boundary_loss_1d(pred_1d_valid)
                        out_of_bounds_1d = ((pred_1d_valid < self.boundary_loss_1d.min_val) |
                                            (pred_1d_valid > self.boundary_loss_1d.max_val)).float().mean()
                    else:
                        boundary_loss_1d = torch.tensor(0.0, device=pred_1d_eval.device, dtype=pred_1d_eval.dtype)
                        out_of_bounds_1d = torch.tensor(0.0, device=pred_1d_eval.device, dtype=pred_1d_eval.dtype)
                    self.log('val/boundary_loss_1d', boundary_loss_1d)
                    self.log('val/out_of_bounds_1d_pct', out_of_bounds_1d * 100)
                if self.boundary_loss_2d is not None:
                    if valid_mask_nodes_2d.any():
                        pred_2d_valid = pred_2d_eval[valid_mask_nodes_2d]
                        boundary_loss_2d = self.boundary_loss_2d(pred_2d_valid)
                        out_of_bounds_2d = ((pred_2d_valid < self.boundary_loss_2d.min_val) |
                                            (pred_2d_valid > self.boundary_loss_2d.max_val)).float().mean()
                    else:
                        boundary_loss_2d = torch.tensor(0.0, device=pred_2d_eval.device, dtype=pred_2d_eval.dtype)
                        out_of_bounds_2d = torch.tensor(0.0, device=pred_2d_eval.device, dtype=pred_2d_eval.dtype)
                    self.log('val/boundary_loss_2d', boundary_loss_2d)
                    self.log('val/out_of_bounds_2d_pct', out_of_bounds_2d * 100)

        self.log('val/rmse_1d', rmse_1d_monitor)
        self.log('val/rmse_2d', rmse_2d_monitor)
        self.log('val/std_rmse', std_rmse_monitor, prog_bar=True)
        # Mirror metric without '/' so checkpoint filenames don't create nested paths.
        self.log('val_std_rmse', std_rmse_monitor, prog_bar=False)

        return {'val_loss': std_rmse_monitor}

    def configure_optimizers(self):
        # Include both model params and trainer-level learnable scalars
        # (e.g., physics source scale calibration terms).
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs, eta_min=self.learning_rate * 0.01)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'}}


# =============================================================================
# DATA MODULE (simplified from dataset.py)
# =============================================================================

class FloodEventDataset(Dataset):
    """Dataset for a single flood event."""

    def __init__(self, data_dir, model_id, event_id, split, graph, seq_len=16, pred_len=90,
                 stride=1, normalize=True, normalization_stats=None, start_only=False):
        self.data_dir = data_dir
        self.model_id = model_id
        self.event_id = event_id
        self.split = split
        self.graph = graph
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stride = stride
        self.normalize = normalize
        self.start_only = start_only

        self._load_dynamic_data()

        if normalization_stats is not None:
            self.norm_stats = normalization_stats
        elif normalize:
            self.norm_stats = self._compute_normalization_stats()
        else:
            self.norm_stats = None

        self._build_sequences()

    def _load_dynamic_data(self):
        event_path = os.path.join(self.data_dir, f"Model_{self.model_id}", self.split, f"event_{self.event_id}")

        df_1d = pd.read_csv(os.path.join(event_path, "1d_nodes_dynamic_all.csv"))
        df_2d = pd.read_csv(os.path.join(event_path, "2d_nodes_dynamic_all.csv"))
        df_ts = pd.read_csv(os.path.join(event_path, "timesteps.csv"))

        self.num_timesteps = len(df_ts)
        self.num_1d_nodes = df_1d['node_idx'].nunique()
        self.num_2d_nodes = df_2d['node_idx'].nunique()

        input_vars_1d = ['water_level', 'inlet_flow']
        input_vars_2d = ['rainfall', 'water_level', 'water_volume']

        self.dynamic_1d = self._reshape_dynamic(df_1d, self.num_1d_nodes, input_vars_1d)
        self.dynamic_2d = self._reshape_dynamic(df_2d, self.num_2d_nodes, input_vars_2d)
        self.water_level_1d = self._reshape_dynamic(df_1d, self.num_1d_nodes, ['water_level'])
        self.water_level_2d = self._reshape_dynamic(df_2d, self.num_2d_nodes, ['water_level'])

    def _reshape_dynamic(self, df, num_nodes, vars):
        available_vars = [v for v in vars if v in df.columns]
        if not available_vars:
            return np.zeros((self.num_timesteps, num_nodes, 1), dtype=np.float32)

        df = df.sort_values(['timestep', 'node_idx'])
        num_timesteps = df['timestep'].nunique()
        data = df[available_vars].values.astype(np.float32)
        data = data.reshape(num_timesteps, num_nodes, len(available_vars))

        for f in range(data.shape[-1]):
            for n in range(data.shape[1]):
                node_vals = data[:, n, f]
                valid_idx = np.where(~np.isnan(node_vals))[0]
                if len(valid_idx) > 0:
                    last_valid_idx = valid_idx[-1]
                    if last_valid_idx < len(node_vals) - 1:
                        data[last_valid_idx + 1:, n, f] = node_vals[last_valid_idx]
                    first_valid_idx = valid_idx[0]
                    if first_valid_idx > 0:
                        data[:first_valid_idx, n, f] = node_vals[first_valid_idx]
                else:
                    data[:, n, f] = 0.0

        return data

    def _compute_normalization_stats(self):
        return {
            '1d': {'mean': self.dynamic_1d.mean(axis=(0, 1)), 'std': self.dynamic_1d.std(axis=(0, 1)) + 1e-8},
            '2d': {'mean': self.dynamic_2d.mean(axis=(0, 1)), 'std': self.dynamic_2d.std(axis=(0, 1)) + 1e-8},
            'target_1d': {'mean': self.water_level_1d.mean(), 'std': self.water_level_1d.std() + 1e-8},
            'target_2d': {'mean': self.water_level_2d.mean(), 'std': self.water_level_2d.std() + 1e-8},
        }

    def _build_sequences(self):
        # Require at least one target timestep after the input prefix.
        max_start = self.num_timesteps - self.seq_len - 1
        if max_start < 0:
            self.valid_starts = []
            return
        if self.start_only:
            self.valid_starts = [0]
            return
        self.valid_starts = list(range(0, max_start + 1, self.stride))

    def __len__(self):
        return len(self.valid_starts)

    def __getitem__(self, idx):
        start = self.valid_starts[idx]
        end_input = start + self.seq_len
        input_1d = self.dynamic_1d[start:end_input].copy()
        input_2d = self.dynamic_2d[start:end_input].copy()

        available_rollout = max(0, self.num_timesteps - end_input)
        rollout_len = min(self.pred_len, available_rollout)

        target_1d = np.zeros((self.pred_len, self.num_1d_nodes), dtype=np.float32)
        target_2d = np.zeros((self.pred_len, self.num_2d_nodes), dtype=np.float32)
        future_rainfall = np.zeros((self.pred_len, self.num_2d_nodes, 1), dtype=np.float32)
        target_mask = np.zeros((self.pred_len,), dtype=np.float32)
        target_mask[:rollout_len] = 1.0

        future_inlet_flow = (
            np.zeros((self.pred_len, self.num_1d_nodes, 1), dtype=np.float32)
            if self.dynamic_1d.shape[-1] > 1 else None
        )

        if rollout_len > 0:
            end_target_valid = end_input + rollout_len
            target_1d[:rollout_len] = self.water_level_1d[end_input:end_target_valid, :, 0]
            target_2d[:rollout_len] = self.water_level_2d[end_input:end_target_valid, :, 0]
            future_rainfall[:rollout_len] = self.dynamic_2d[end_input:end_target_valid, :, 0:1]
            if future_inlet_flow is not None:
                future_inlet_flow[:rollout_len] = self.dynamic_1d[end_input:end_target_valid, :, 1:2]

        if self.normalize and self.norm_stats is not None:
            input_1d = (input_1d - self.norm_stats['1d']['mean']) / self.norm_stats['1d']['std']
            input_2d = (input_2d - self.norm_stats['2d']['mean']) / self.norm_stats['2d']['std']
            target_1d = (target_1d - self.norm_stats['target_1d']['mean']) / self.norm_stats['target_1d']['std']
            target_2d = (target_2d - self.norm_stats['target_2d']['mean']) / self.norm_stats['target_2d']['std']
            future_rainfall = (future_rainfall - self.norm_stats['2d']['mean'][0]) / self.norm_stats['2d']['std'][0]
            if future_inlet_flow is not None:
                future_inlet_flow = (future_inlet_flow - self.norm_stats['1d']['mean'][1]) / self.norm_stats['1d']['std'][1]

        result = {
            'input_1d': torch.from_numpy(input_1d),
            'input_2d': torch.from_numpy(input_2d),
            'target_1d': torch.from_numpy(target_1d),
            'target_2d': torch.from_numpy(target_2d),
            'target_mask': torch.from_numpy(target_mask),
            'future_rainfall': torch.from_numpy(future_rainfall),
            'prefix_len': self.seq_len,
            'rollout_len': rollout_len,
        }

        if future_inlet_flow is not None:
            result['future_inlet_flow'] = torch.from_numpy(future_inlet_flow)

        return result


class FloodDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for flood modelling."""

    def __init__(self, data_dir, model_id, batch_size=4, seq_len=10, pred_len=90, stride=4,
                 val_ratio=0.2, num_workers=4, graph_kwargs=None,
                 train_start_only=False, val_start_only=False):
        super().__init__()
        self.data_dir = data_dir
        self.model_id = model_id
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.stride = stride
        self.val_ratio = val_ratio
        self.num_workers = num_workers
        self.graph_kwargs = graph_kwargs or {}
        self.train_start_only = train_start_only
        self.val_start_only = val_start_only

        self.graph = None
        self.train_datasets = []
        self.val_datasets = []
        self.norm_stats = None

    def setup(self, stage=None):
        builder = FloodGraphBuilder(self.data_dir, self.model_id, **self.graph_kwargs)
        self.graph = builder.build(split="train")

        train_path = os.path.join(self.data_dir, f"Model_{self.model_id}", "train")
        train_events = self._discover_events(train_path)

        np.random.seed(42)
        np.random.shuffle(train_events)
        val_size = int(len(train_events) * self.val_ratio)
        val_events = train_events[:val_size]
        train_events = train_events[val_size:]

        print(f"Model {self.model_id}: {len(train_events)} train, {len(val_events)} val events")

        self.norm_stats = self._compute_global_norm_stats(train_events, "train")

        if stage == "fit" or stage is None:
            self.train_datasets = [
                FloodEventDataset(
                    self.data_dir, self.model_id, event_id, "train", self.graph,
                    self.seq_len, self.pred_len, self.stride, normalization_stats=self.norm_stats,
                    start_only=self.train_start_only,
                )
                for event_id in train_events
            ]
            self.val_datasets = [
                FloodEventDataset(
                    self.data_dir, self.model_id, event_id, "train", self.graph,
                    self.seq_len, self.pred_len, self.stride, normalization_stats=self.norm_stats,
                    start_only=self.val_start_only,
                )
                for event_id in val_events
            ]

    def _discover_events(self, path):
        events = []
        if os.path.exists(path):
            for item in os.listdir(path):
                if item.startswith("event_"):
                    try:
                        event_id = int(item.split("_")[1])
                        events.append(event_id)
                    except ValueError:
                        pass
        return sorted(events)

    def _compute_global_norm_stats(self, event_ids, split):
        all_1d, all_2d, all_wl_1d, all_wl_2d = [], [], [], []

        for event_id in event_ids:
            ds = FloodEventDataset(
                self.data_dir, self.model_id, event_id, split, self.graph,
                self.seq_len, self.pred_len, self.stride, normalize=False
            )
            all_1d.append(ds.dynamic_1d)
            all_2d.append(ds.dynamic_2d)
            all_wl_1d.append(ds.water_level_1d)
            all_wl_2d.append(ds.water_level_2d)

        all_1d = np.concatenate(all_1d, axis=0)
        all_2d = np.concatenate(all_2d, axis=0)
        all_wl_1d = np.concatenate(all_wl_1d, axis=0)
        all_wl_2d = np.concatenate(all_wl_2d, axis=0)

        return {
            '1d': {'mean': all_1d.mean(axis=(0, 1)), 'std': all_1d.std(axis=(0, 1)) + 1e-8},
            '2d': {'mean': all_2d.mean(axis=(0, 1)), 'std': all_2d.std(axis=(0, 1)) + 1e-8},
            'target_1d': {'mean': all_wl_1d.mean(), 'std': all_wl_1d.std() + 1e-8},
            'target_2d': {'mean': all_wl_2d.mean(), 'std': all_wl_2d.std() + 1e-8},
        }

    def train_dataloader(self):
        combined = torch.utils.data.ConcatDataset(self.train_datasets)
        return DataLoader(
            combined, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=4 if self.num_workers > 0 else None,
        )

    def val_dataloader(self):
        combined = torch.utils.data.ConcatDataset(self.val_datasets)
        return DataLoader(
            combined, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=True if self.num_workers > 0 else False,
            prefetch_factor=4 if self.num_workers > 0 else None,
        )


# =============================================================================
# NORMALIZATION HELPERS
# =============================================================================

def split_train_val_events(data_dir: str, model_id: int, val_ratio: float = 0.2, seed: int = 42) -> Tuple[List[int], List[int]]:
    """Reproduce the exact train/val event split used during training."""
    train_path = os.path.join(data_dir, f"Model_{model_id}", "train")
    events = []
    if os.path.exists(train_path):
        for item in os.listdir(train_path):
            if item.startswith("event_"):
                try:
                    events.append(int(item.split("_")[1]))
                except ValueError:
                    pass
    events = sorted(events)

    shuffled = np.array(events, dtype=np.int64)
    np.random.seed(seed)
    np.random.shuffle(shuffled)
    val_size = int(len(shuffled) * val_ratio)

    val_events = shuffled[:val_size].tolist()
    train_events = shuffled[val_size:].tolist()
    return train_events, val_events


def infer_competition_max_horizon(
    data_dir: str,
    model_id: int,
    sample_submission_path: Optional[str] = None,
) -> Optional[int]:
    """Infer required max future rollout steps from sample_submission for a model.

    Returns:
        Maximum number of rows per node (i.e., required forecast steps) for the
        provided model_id, or None if sample submission is unavailable.
    """
    sample_path = sample_submission_path or os.path.join(data_dir, "sample_submission.parquet")
    if not os.path.exists(sample_path):
        return None

    try:
        sample = pd.read_parquet(
            sample_path,
            columns=["model_id", "event_id", "node_type", "node_id"],
        )
    except Exception:
        return None

    sample = sample[sample["model_id"] == int(model_id)]
    if sample.empty:
        return None

    per_event_horizons: List[int] = []
    for _, event_df in sample.groupby("event_id", sort=False):
        node_type_steps: List[int] = []
        for node_type in (1, 2):
            node_df = event_df[event_df["node_type"] == node_type]
            if node_df.empty:
                continue
            num_nodes = int(node_df["node_id"].nunique())
            if num_nodes <= 0:
                continue
            steps = int(len(node_df) // num_nodes)
            node_type_steps.append(steps)
        if node_type_steps:
            per_event_horizons.append(max(node_type_steps))

    if not per_event_horizons:
        return None
    return int(max(per_event_horizons))


def compute_norm_stats_for_events(
    data_dir: str,
    model_id: int,
    event_ids: List[int],
    graph,
    seq_len: int,
    pred_len: int,
    stride: int,
) -> Dict[str, Dict[str, np.ndarray]]:
    """Compute normalization stats from specific training events."""
    if not event_ids:
        raise ValueError(f"No events provided for norm stats (model_id={model_id})")

    sum_1d = np.zeros(2, dtype=np.float64)
    sq_sum_1d = np.zeros(2, dtype=np.float64)
    count_1d = 0

    sum_2d = np.zeros(3, dtype=np.float64)
    sq_sum_2d = np.zeros(3, dtype=np.float64)
    count_2d = 0

    sum_t1 = 0.0
    sq_sum_t1 = 0.0
    count_t1 = 0

    sum_t2 = 0.0
    sq_sum_t2 = 0.0
    count_t2 = 0

    for event_id in event_ids:
        ds = FloodEventDataset(
            data_dir, model_id, event_id, "train", graph,
            seq_len=seq_len, pred_len=pred_len, stride=stride, normalize=False
        )

        dyn_1d = ds.dynamic_1d.reshape(-1, ds.dynamic_1d.shape[-1]).astype(np.float64, copy=False)
        dyn_2d = ds.dynamic_2d.reshape(-1, ds.dynamic_2d.shape[-1]).astype(np.float64, copy=False)
        wl_1d = ds.water_level_1d.reshape(-1).astype(np.float64, copy=False)
        wl_2d = ds.water_level_2d.reshape(-1).astype(np.float64, copy=False)

        sum_1d += dyn_1d.sum(axis=0)
        sq_sum_1d += np.square(dyn_1d).sum(axis=0)
        count_1d += dyn_1d.shape[0]

        sum_2d += dyn_2d.sum(axis=0)
        sq_sum_2d += np.square(dyn_2d).sum(axis=0)
        count_2d += dyn_2d.shape[0]

        sum_t1 += float(wl_1d.sum())
        sq_sum_t1 += float(np.square(wl_1d).sum())
        count_t1 += wl_1d.size

        sum_t2 += float(wl_2d.sum())
        sq_sum_t2 += float(np.square(wl_2d).sum())
        count_t2 += wl_2d.size

    if min(count_1d, count_2d, count_t1, count_t2) == 0:
        raise RuntimeError("Encountered zero-sized accumulators while computing normalization stats")

    mean_1d = sum_1d / count_1d
    mean_2d = sum_2d / count_2d
    mean_t1 = sum_t1 / count_t1
    mean_t2 = sum_t2 / count_t2

    var_1d = np.maximum(sq_sum_1d / count_1d - np.square(mean_1d), 1e-12)
    var_2d = np.maximum(sq_sum_2d / count_2d - np.square(mean_2d), 1e-12)
    var_t1 = max(sq_sum_t1 / count_t1 - mean_t1 ** 2, 1e-12)
    var_t2 = max(sq_sum_t2 / count_t2 - mean_t2 ** 2, 1e-12)

    return {
        '1d': {'mean': mean_1d.astype(np.float32), 'std': (np.sqrt(var_1d) + 1e-8).astype(np.float32)},
        '2d': {'mean': mean_2d.astype(np.float32), 'std': (np.sqrt(var_2d) + 1e-8).astype(np.float32)},
        'target_1d': {'mean': float(mean_t1), 'std': float(np.sqrt(var_t1) + 1e-8)},
        'target_2d': {'mean': float(mean_t2), 'std': float(np.sqrt(var_t2) + 1e-8)},
    }


# =============================================================================
# MAIN TRAINING SCRIPT
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description='Train Physics-Informed VGSSM model (standalone)')

    # Mode
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict'],
                        help='Mode: train or predict')

    # Data
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--model_id', type=int, required=True, choices=[1, 2])

    # Model architecture
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--event_latent_dim', type=int, default=16)
    parser.add_argument('--num_gnn_layers', type=int, default=3)
    parser.add_argument('--num_transition_gnn_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--latent_sample_temperature', type=float, default=1.0,
                        help='Scale factor for posterior latent sampling noise')
    parser.add_argument('--latent_state_clip', type=float, default=10.0,
                        help='Clamp latent state magnitude; <=0 disables clipping')

    # Prediction
    parser.add_argument('--prediction_horizon', type=int, default=90)
    parser.add_argument('--auto_prediction_horizon', action='store_true',
                        help='Auto-set prediction_horizon from sample_submission for selected model')
    parser.add_argument('--strict_horizon_check', action='store_true',
                        help='Fail if prediction_horizon is below required sample_submission horizon')
    parser.add_argument('--sample_submission_path', type=str, default=None,
                        help='Optional path to sample_submission.parquet (defaults to <data_dir>/sample_submission.parquet)')
    parser.add_argument('--prefix_len', type=int, default=10)

    # Training
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--beta_ce', type=float, default=0.01)
    parser.add_argument('--beta_z', type=float, default=0.001)
    parser.add_argument('--horizon_weighting', type=str, default='linear')
    parser.add_argument('--recon_balance_mode', type=str, default='equal', choices=['equal', 'sum'],
                        help='How to combine 1D/2D reconstruction losses: equal (metric-aligned) or sum (legacy)')
    parser.add_argument('--recon_weight_1d', type=float, default=0.5,
                        help='1D reconstruction weight when recon_balance_mode=equal (renormalized with 2D)')
    parser.add_argument('--recon_weight_2d', type=float, default=0.5,
                        help='2D reconstruction weight when recon_balance_mode=equal (renormalized with 1D)')
    parser.add_argument('--horizon_weight_by_valid_count', action='store_true',
                        help='Scale horizon weights by number of valid samples at each step (reduces tail-noise over-weighting)')
    parser.add_argument('--loss_type', type=str, default='mse')
    parser.add_argument('--warmup_epochs', type=int, default=5)

    parser.add_argument('--seq_len', type=int, default=10)
    parser.add_argument('--stride', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--train_start_only', action='store_true',
                        help='Use only start=0 windows for training')
    parser.add_argument('--val_start_only', action='store_true',
                        help='Use only start=0 windows for validation')
    parser.add_argument('--future_inlet_mode_train', type=str, default='missing',
                        choices=['full', 'missing', 'mixed'],
                        help='Future inlet flow mode during training: full, missing, or mixed')
    parser.add_argument('--future_inlet_keep_prob', type=float, default=0.0,
                        help='When future_inlet_mode_train=mixed, probability of keeping future inlet input')

    parser.add_argument('--exp_name', type=str, default='vgssm_physics')
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--accumulate_grad_batches', type=int, default=1)
    parser.add_argument('--limit_val_batches', type=float, default=1.0,
                        help='Fraction or count of validation batches to run each validation epoch')
    parser.add_argument('--limit_train_batches', type=float, default=1.0,
                        help='Fraction or count of training batches to run each training epoch')
    parser.add_argument('--num_sanity_val_steps', type=int, default=2,
                        help='Number of sanity validation batches before training')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1,
                        help='Run validation every N epochs')
    parser.add_argument('--precision', type=str, default='32', choices=['32', '16', 'bf16'],
                        help='Training precision: 32 (FP32), 16 (FP16 mixed), bf16 (bfloat16)')

    parser.add_argument('--accelerator', type=str, default='auto')
    parser.add_argument('--devices', type=int, default=1)

    # Physics-informed settings
    parser.add_argument('--use_physics_loss', action='store_true',
                        help='Enable physics conservation losses')
    parser.add_argument('--physics_local_weight', type=float, default=0.1,
                        help='Weight for local mass conservation loss')
    parser.add_argument('--physics_global_weight', type=float, default=0.01,
                        help='Weight for global mass conservation loss')
    parser.add_argument('--physics_loss_mode', type=str, default='residual', choices=['residual', 'light', 'legacy'],
                        help='Physics loss type: residual (recommended), light regularizer, or legacy flow-based')
    parser.add_argument('--physics_residual_huber_delta', type=float, default=0.1,
                        help='Huber delta used in residual physics loss normalization')
    parser.add_argument('--use_delta_prediction', action='store_true',
                        help='Predict deltas instead of absolute values')
    parser.add_argument('--dt_seconds', type=float, default=300.0,
                        help='Timestep in seconds for dV/dt computation')
    parser.add_argument('--physics_subsample_rate', type=int, default=5,
                        help='Compute physics loss every N timesteps (memory optimization, default=5)')

    # Output bounds for water level predictions
    parser.add_argument('--output_bounds_1d_min', type=float, default=None,
                        help='Min water level for 1D nodes (auto-detected if not specified)')
    parser.add_argument('--output_bounds_1d_max', type=float, default=None,
                        help='Max water level for 1D nodes (auto-detected if not specified)')
    parser.add_argument('--output_bounds_2d_min', type=float, default=None,
                        help='Min water level for 2D nodes (auto-detected if not specified)')
    parser.add_argument('--output_bounds_2d_max', type=float, default=None,
                        help='Max water level for 2D nodes (auto-detected if not specified)')
    parser.add_argument('--boundary_loss_weight', type=float, default=0.1,
                        help='Weight for soft boundary loss penalty (default: 0.1)')
    parser.add_argument('--boundary_delta', type=float, default=10.0,
                        help='Huber delta for boundary loss - quadratic below, linear above (default: 10.0)')
    parser.add_argument('--no_boundary_loss', action='store_true',
                        help='Disable soft boundary loss')
    parser.add_argument('--use_baseline_residual', action='store_true',
                        help='Predict residuals from per-node mean (better bounded outputs)')
    parser.add_argument('--use_sigmoid_bounds', action='store_true',
                        help='Use sigmoid scaling for bounded outputs: h = min + sigmoid(x) * (max - min)')
    parser.add_argument('--use_physics_decoder', action='store_true',
                        help='Use physics-based decoder: K -> Q -> V -> h (mass-conserving, naturally bounded)')
    parser.add_argument('--physics_dt', type=float, default=300.0,
                        help='Time step for physics decoder [seconds] (default: 300)')

    # Curriculum learning settings
    parser.add_argument('--use_curriculum', action='store_true',
                        help='Enable curriculum learning with progressive rollout')
    parser.add_argument('--curriculum_stages', type=str, default='1,4,8,16,32,90',
                        help='Comma-separated rollout lengths for curriculum')
    parser.add_argument('--epochs_per_stage', type=int, default=5,
                        help='Minimum epochs per curriculum stage')
    parser.add_argument('--patience_per_stage', type=int, default=3,
                        help='Patience for curriculum stage advancement')

    # Timer settings (causal Transformer for temporal prior)
    parser.add_argument('--use_timer', action='store_true',
                        help='Enable Timer v3 (DEPRECATED: has fundamental issues)')
    parser.add_argument('--use_timer_v4', action='store_true',
                        help='Enable Timer v4 (RECOMMENDED: bidirectional attention + mean pooling)')
    parser.add_argument('--timer_v4_pooling', type=str, default='mean', choices=['mean', 'max', 'attention', 'last'],
                        help='Pooling method for Timer v4 (mean recommended)')
    parser.add_argument('--timer_layers', type=int, default=4,
                        help='Number of Timer Transformer layers')
    parser.add_argument('--timer_heads', type=int, default=4,
                        help='Number of attention heads in Timer')
    parser.add_argument('--timer_history_len', type=int, default=10,
                        help='History length for Timer temporal context')
    parser.add_argument('--timer_transition_variant', type=str, default='auto', choices=['auto', 'v3', 'v5'],
                        help='Transition temporal prior variant: auto chooses v5 with --use_timer_v4 else v3')
    parser.add_argument('--timer_enable_2d_context', action='store_true',
                        help='Enable low-cost Timer-derived global temporal context for 2D nodes')

    # Grassmann Flow settings (attention-free alternative)
    parser.add_argument('--use_grassmann', action='store_true',
                        help='Use Grassmann Flow posterior (attention-free, linear complexity)')
    parser.add_argument('--grassmann_layers', type=int, default=4,
                        help='Number of Grassmann mixing blocks')
    parser.add_argument('--grassmann_rank', type=int, default=12,
                        help='Reduced rank for Plücker encoding (Plücker dim = r*(r-1)/2)')
    parser.add_argument('--grassmann_offsets', type=str, default='1,2,4,8,16,32',
                        help='Comma-separated multi-scale offsets for temporal pairing')

    # Physics-constrained transition (blends neural + physics updates)
    parser.add_argument('--use_physics_transition', action='store_true',
                        help='Use PhysicsConstrainedTransition (neural + physics delta blending)')

    # Prediction specific
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint for prediction')
    parser.add_argument('--init_ckpt', type=str, default=None,
                        help='Path to initialization checkpoint for training (loads model weights only)')
    parser.add_argument('--calibrate_latent', action='store_true',
                        help='Enable test-time latent calibration')
    parser.add_argument('--calibration_steps', type=int, default=50,
                        help='Number of calibration steps')
    parser.add_argument('--predict_chunk_size', type=int, default=0,
                        help='Chunk size for prediction rollout. <=0 uses min(prediction_horizon, needed_steps)')
    parser.add_argument('--predict_rollout_stateful', action='store_true',
                        help='Carry latent state across rollout chunks in prediction (default: reset per chunk)')
    parser.add_argument('--predict_stochastic_latent', action='store_true',
                        help='Use stochastic latent sampling in prediction (default uses posterior means)')
    parser.add_argument('--predict_max_ckpt_std_rmse', type=float, default=0.10,
                        help='Abort prediction if checkpoint best val/std_rmse exceeds this (<=0 disables)')
    parser.add_argument('--predict_max_std_ratio', type=float, default=2.5,
                        help='Abort prediction if output std exceeds this x training target std (<=0 disables)')

    return parser.parse_args()


def get_model_specific_config(model_id, args):
    if model_id == 2:
        return {
            'hidden_dim': max(args.hidden_dim, 96),
            'latent_dim': max(args.latent_dim, 48),
            'num_gnn_layers': max(args.num_gnn_layers, 4),
            'num_transition_gnn_layers': max(args.num_transition_gnn_layers, 3),
            'dropout': min(args.dropout, 0.15),
        }
    return {
        'hidden_dim': args.hidden_dim,
        'latent_dim': args.latent_dim,
        'num_gnn_layers': args.num_gnn_layers,
        'num_transition_gnn_layers': args.num_transition_gnn_layers,
        'dropout': args.dropout,
    }


def resolve_physical_output_bounds(model_id: int, args) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Resolve default or user-provided water level bounds in physical units."""
    if args.output_bounds_1d_min is not None and args.output_bounds_1d_max is not None:
        bounds_1d = (float(args.output_bounds_1d_min), float(args.output_bounds_1d_max))
    elif model_id == 1:
        bounds_1d = (280.0, 370.0)  # Model 1 observed: 286-362
    else:
        bounds_1d = (15.0, 70.0)    # Model 2 observed: 22-60

    if args.output_bounds_2d_min is not None and args.output_bounds_2d_max is not None:
        bounds_2d = (float(args.output_bounds_2d_min), float(args.output_bounds_2d_max))
    elif model_id == 1:
        bounds_2d = (280.0, 370.0)  # Model 1 observed: 286-362
    else:
        bounds_2d = (15.0, 70.0)    # Model 2 observed: 22-60

    return bounds_1d, bounds_2d


def convert_bounds_to_normalized(
    bounds: Tuple[float, float],
    mean: float,
    std: float,
) -> Tuple[float, float]:
    """Convert physical bounds into normalized target space."""
    std = max(float(std), 1e-8)
    mean = float(mean)
    return ((float(bounds[0]) - mean) / std, (float(bounds[1]) - mean) / std)


def resolve_internal_output_bounds(
    model_id: int,
    args,
    norm_stats: Optional[Dict[str, Dict[str, float]]],
) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    """Return physical bounds and model-internal bounds (normalized if stats available)."""
    physical_1d, physical_2d = resolve_physical_output_bounds(model_id, args)
    internal_1d, internal_2d = physical_1d, physical_2d

    if norm_stats is not None:
        internal_1d = convert_bounds_to_normalized(
            physical_1d,
            norm_stats['target_1d']['mean'],
            norm_stats['target_1d']['std'],
        )
        internal_2d = convert_bounds_to_normalized(
            physical_2d,
            norm_stats['target_2d']['mean'],
            norm_stats['target_2d']['std'],
        )

    return physical_1d, physical_2d, internal_1d, internal_2d


def strip_model_prefix_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Normalize checkpoint keys by removing optional 'model.' prefix."""
    normalized = {}
    for key, value in state_dict.items():
        if key.startswith('model.'):
            normalized[key[6:]] = value
        else:
            normalized[key] = value
    return normalized


def infer_model_config_from_state_dict(
    state_dict: Dict[str, torch.Tensor],
    fallback_model_config: Dict[str, Any],
    fallback_event_latent_dim: int,
    fallback_num_heads: int,
) -> Dict[str, Any]:
    """Infer core architecture dimensions directly from checkpoint tensors."""
    inferred = dict(fallback_model_config)
    inferred['event_latent_dim'] = int(fallback_event_latent_dim)
    inferred['num_heads'] = int(fallback_num_heads)

    key_hidden = 'spatial_encoder.gnn.input_proj.1d.weight'
    if key_hidden in state_dict:
        inferred['hidden_dim'] = int(state_dict[key_hidden].shape[0])

    key_latent = 'z0_prior_mean'
    if key_latent in state_dict:
        inferred['latent_dim'] = int(state_dict[key_latent].numel())

    key_event = 'event_encoder.mean_proj.weight'
    if key_event in state_dict:
        inferred['event_latent_dim'] = int(state_dict[key_event].shape[0])

    conv_idx = []
    conv_pattern = re.compile(r'^spatial_encoder\.gnn\.convs\.(\d+)\.')
    for key in state_dict.keys():
        m = conv_pattern.match(key)
        if m:
            conv_idx.append(int(m.group(1)))
    if conv_idx:
        inferred['num_gnn_layers'] = max(conv_idx) + 1

    trans_idx = []
    trans_pattern = re.compile(r'^transition\.gnn_blocks\.(\d+)\.')
    for key in state_dict.keys():
        m = trans_pattern.match(key)
        if m:
            trans_idx.append(int(m.group(1)))
    if trans_idx:
        inferred['num_transition_gnn_layers'] = max(trans_idx) + 1

    attn_key = None
    for key in state_dict.keys():
        if key.endswith('.att_src'):
            attn_key = key
            break
    if attn_key is not None and state_dict[attn_key].ndim >= 2:
        inferred['num_heads'] = int(state_dict[attn_key].shape[1])

    return inferred


def infer_model_config_from_checkpoint_path(
    ckpt_path: Optional[str],
    fallback_model_config: Dict[str, Any],
    fallback_event_latent_dim: int,
    fallback_num_heads: int,
) -> Optional[Dict[str, Any]]:
    """Infer architecture from a checkpoint file if available."""
    if not ckpt_path:
        return None
    if not os.path.exists(ckpt_path):
        return None
    try:
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        state_dict = checkpoint.get('state_dict')
        if not isinstance(state_dict, dict):
            return None
        normalized_state = strip_model_prefix_from_state_dict(state_dict)
        return infer_model_config_from_state_dict(
            state_dict=normalized_state,
            fallback_model_config=fallback_model_config,
            fallback_event_latent_dim=fallback_event_latent_dim,
            fallback_num_heads=fallback_num_heads,
        )
    except Exception as exc:
        print(f"Warning: failed to infer architecture from checkpoint '{ckpt_path}': {exc}")
        return None


def main():
    args = parse_args()

    required_horizon = infer_competition_max_horizon(
        data_dir=args.data_dir,
        model_id=args.model_id,
        sample_submission_path=args.sample_submission_path,
    )
    if required_horizon is not None:
        print(f"Detected competition max horizon from sample submission: {required_horizon}")
        if args.auto_prediction_horizon and int(args.prediction_horizon) != int(required_horizon):
            print(
                f"Auto horizon enabled: overriding prediction_horizon "
                f"{args.prediction_horizon} -> {required_horizon}"
            )
            args.prediction_horizon = int(required_horizon)
        elif int(args.prediction_horizon) < int(required_horizon):
            msg = (
                f"prediction_horizon={args.prediction_horizon} is below required "
                f"competition horizon={required_horizon}. Long-rollout performance may degrade."
            )
            if args.strict_horizon_check and args.mode == 'train':
                raise ValueError(msg)
            print(f"WARNING: {msg}")

    if torch.cuda.is_available():
        # Improves Tensor Core utilization for FP32 matmuls on A100-class GPUs.
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = True

    if args.mode == 'predict':
        if args.checkpoint is None:
            args.checkpoint = f'checkpoints/model_{args.model_id}/vgssm_physics/best.ckpt'
        generate_submission(args)
    else:
        # Training mode
        pl.seed_everything(42, workers=True)

        model_config = get_model_specific_config(args.model_id, args)
        arch_ckpt_path = None
        if args.checkpoint and os.path.exists(args.checkpoint):
            arch_ckpt_path = args.checkpoint
        elif args.init_ckpt and os.path.exists(args.init_ckpt):
            arch_ckpt_path = args.init_ckpt

        inferred_config = infer_model_config_from_checkpoint_path(
            ckpt_path=arch_ckpt_path,
            fallback_model_config=model_config,
            fallback_event_latent_dim=args.event_latent_dim,
            fallback_num_heads=args.num_heads,
        )
        if inferred_config is not None:
            model_config['hidden_dim'] = inferred_config['hidden_dim']
            model_config['latent_dim'] = inferred_config['latent_dim']
            model_config['num_gnn_layers'] = inferred_config['num_gnn_layers']
            model_config['num_transition_gnn_layers'] = inferred_config['num_transition_gnn_layers']
            model_config['dropout'] = inferred_config['dropout']
            args.event_latent_dim = inferred_config['event_latent_dim']
            args.num_heads = inferred_config['num_heads']
            print(
                "Architecture inferred from checkpoint "
                f"({arch_ckpt_path}): "
                f"hidden_dim={model_config['hidden_dim']}, "
                f"latent_dim={model_config['latent_dim']}, "
                f"gnn_layers={model_config['num_gnn_layers']}, "
                f"transition_gnn={model_config['num_transition_gnn_layers']}, "
                f"event_latent_dim={args.event_latent_dim}, "
                f"num_heads={args.num_heads}"
            )

        print("=" * 70)
        print(f"Training Physics-Informed VGSSM for Model {args.model_id}")
        print(f"Experiment: {args.exp_name}")
        print(f"Config: hidden_dim={model_config['hidden_dim']}, latent_dim={model_config['latent_dim']}")
        print(f"        gnn_layers={model_config['num_gnn_layers']}, transition_gnn={model_config['num_transition_gnn_layers']}")
        print(f"Physics: delta={args.use_delta_prediction}, physics_loss={args.use_physics_loss}, curriculum={args.use_curriculum}")
        if args.use_physics_loss:
            print(
                f"         mode={args.physics_loss_mode}, local_weight={args.physics_local_weight}, "
                f"global_weight={args.physics_global_weight}, subsample_rate={args.physics_subsample_rate}"
            )
        if args.use_curriculum:
            print(f"         stages={args.curriculum_stages}, epochs_per_stage={args.epochs_per_stage}")
        print(f"Latent: sample_temp={args.latent_sample_temperature}, state_clip={args.latent_state_clip}")
        print(f"Future inlet (train): mode={args.future_inlet_mode_train}, keep_prob={args.future_inlet_keep_prob}")
        print(f"Recon: balance={args.recon_balance_mode} (w1d={args.recon_weight_1d}, w2d={args.recon_weight_2d}), horizon_valid_count={args.horizon_weight_by_valid_count}")
        print(f"Windowing: train_start_only={args.train_start_only}, val_start_only={args.val_start_only}")
        print(f"Validation: limit_val_batches={args.limit_val_batches}, check_every_n_epoch={args.check_val_every_n_epoch}, sanity_steps={args.num_sanity_val_steps}")
        use_timer = getattr(args, 'use_timer', False)
        use_timer_v4 = getattr(args, 'use_timer_v4', False)
        print(f"Timer: v3={use_timer}, v4={use_timer_v4}")
        if use_timer or use_timer_v4:
            print(f"       layers={args.timer_layers}, heads={args.timer_heads}")
            print(
                f"       transition_variant={getattr(args, 'timer_transition_variant', 'auto')}, "
                f"2d_context={bool(getattr(args, 'timer_enable_2d_context', False))}"
            )
            if use_timer_v4:
                print(f"       v4_pooling={getattr(args, 'timer_v4_pooling', 'mean')}")
        print("=" * 70)

        if torch.cuda.is_available():
            print(f"CUDA: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            print("MPS available")
        else:
            print("CPU only")

        print("\n[1/4] Loading data...")
        data_module = FloodDataModule(
            data_dir=args.data_dir,
            model_id=args.model_id,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            pred_len=args.prediction_horizon,
            stride=args.stride,
            num_workers=args.num_workers,
            graph_kwargs={'add_knn_2d_edges': True, 'knn_k': 8},
            train_start_only=args.train_start_only,
            val_start_only=args.val_start_only,
        )
        data_module.setup('fit')

        norm_stats = data_module.norm_stats
        graph = data_module.graph
        print(f"  Training samples: {sum(len(ds) for ds in data_module.train_datasets)}")
        print(f"  Validation samples: {sum(len(ds) for ds in data_module.val_datasets)}")

        print("\n[2/4] Building graph...")
        static_1d_dim = graph['1d'].x.shape[1]
        static_2d_dim = graph['2d'].x.shape[1]
        num_1d_nodes = graph['1d'].x.shape[0]
        num_2d_nodes = graph['2d'].x.shape[0]
        print(f"  1D nodes: {num_1d_nodes}, features: {static_1d_dim}")
        print(f"  2D nodes: {num_2d_nodes}, features: {static_2d_dim}")

        dynamic_1d_dim = 2
        dynamic_2d_dim = 3

        device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
        graph = graph.to(device)

        print("\n[3/4] Creating Physics-Informed VGSSM model...")

        # IMPORTANT: targets are normalized in the dataset, so model-internal
        # bounds must also be in normalized space to avoid loss-scale mismatch.
        (
            output_bounds_1d_physical,
            output_bounds_2d_physical,
            output_bounds_1d,
            output_bounds_2d,
        ) = resolve_internal_output_bounds(args.model_id, args, norm_stats)

        print(f"  Output bounds 1D (physical): {output_bounds_1d_physical}")
        print(f"  Output bounds 2D (physical): {output_bounds_2d_physical}")
        if norm_stats is not None:
            print(f"  Output bounds 1D (normalized): {output_bounds_1d}")
            print(f"  Output bounds 2D (normalized): {output_bounds_2d}")
        print(f"  Soft boundary loss: {'enabled' if not args.no_boundary_loss else 'disabled'}")
        print(f"  Boundary loss weight: {args.boundary_loss_weight}")
        use_sigmoid_bounds = getattr(args, 'use_sigmoid_bounds', False)
        if use_sigmoid_bounds:
            print(f"  Sigmoid bounds: ENABLED (h = min + sigmoid(x) * (max - min))")

        use_timer = getattr(args, 'use_timer', False)
        use_timer_v4 = getattr(args, 'use_timer_v4', False)
        timer_v4_pooling = getattr(args, 'timer_v4_pooling', 'mean')
        use_grassmann = getattr(args, 'use_grassmann', False)
        grassmann_offsets = None
        if hasattr(args, 'grassmann_offsets') and args.grassmann_offsets:
            grassmann_offsets = [int(x) for x in args.grassmann_offsets.split(',')]

        model = VGSSM(
            static_1d_dim=static_1d_dim,
            static_2d_dim=static_2d_dim,
            dynamic_1d_dim=dynamic_1d_dim,
            dynamic_2d_dim=dynamic_2d_dim,
            hidden_dim=model_config['hidden_dim'],
            latent_dim=model_config['latent_dim'],
            event_latent_dim=args.event_latent_dim,
            num_gnn_layers=model_config['num_gnn_layers'],
            num_transition_gnn_layers=model_config['num_transition_gnn_layers'],
            num_heads=args.num_heads,
            prediction_horizon=args.prediction_horizon,
            use_event_latent=True,
            dropout=model_config['dropout'],
            use_delta_prediction=args.use_delta_prediction,
            use_physics_loss=args.use_physics_loss,
            physics_subsample_rate=args.physics_subsample_rate,  # Memory optimization
            # Timer settings
            use_timer=use_timer,
            timer_layers=getattr(args, 'timer_layers', 4),
            timer_heads=getattr(args, 'timer_heads', 4),
            timer_history_len=getattr(args, 'timer_history_len', 10),
            # Timer V4 settings (scientific fix)
            use_timer_v4=use_timer_v4,
            timer_v4_pooling=timer_v4_pooling,
            timer_transition_variant=getattr(args, 'timer_transition_variant', 'auto'),
            timer_enable_2d_context=bool(getattr(args, 'timer_enable_2d_context', False)),
            # Grassmann Flow settings (attention-free)
            use_grassmann=use_grassmann,
            grassmann_layers=getattr(args, 'grassmann_layers', 4),
            grassmann_rank=getattr(args, 'grassmann_rank', 12),
            grassmann_offsets=grassmann_offsets,
            # Physics-constrained transition
            use_physics_transition=getattr(args, 'use_physics_transition', False),
            # Output bounds to prevent invalid predictions
            output_bounds_1d=output_bounds_1d,
            output_bounds_2d=output_bounds_2d,
            # Baseline residual mode for better bounded outputs
            use_baseline_residual=getattr(args, 'use_baseline_residual', False),
            num_1d_nodes=graph['1d'].num_nodes,
            num_2d_nodes=graph['2d'].num_nodes,
            # Sigmoid bounds: h = min + sigmoid(x) * (max - min)
            use_sigmoid_bounds=getattr(args, 'use_sigmoid_bounds', False),
            # Physics decoder: K -> Q -> V -> h (mass-conserving)
            use_physics_decoder=getattr(args, 'use_physics_decoder', False),
            node_areas_1d=None,  # Default 1 m² per node
            node_areas_2d=None,  # Default 1 m² per node
            physics_dt=getattr(args, 'physics_dt', 300.0),
            latent_sample_temperature=args.latent_sample_temperature,
            latent_state_clip=args.latent_state_clip,
        )

        # Print physics decoder status
        if getattr(args, 'use_physics_decoder', False):
            print(f"  Physics decoder: ENABLED (mass-conserving K->Q->V->h)")
            print(f"  Physics dt: {args.physics_dt} seconds")

        # If using baseline residual, compute per-node mean from training data
        if getattr(args, 'use_baseline_residual', False):
            print("  Computing per-node baseline from training data...")
            baseline_1d_physical, baseline_2d_physical = compute_per_node_baseline(
                args.data_dir, args.model_id, graph
            )
            if norm_stats is not None:
                mean_1d = float(norm_stats['target_1d']['mean'])
                std_1d = max(float(norm_stats['target_1d']['std']), 1e-8)
                mean_2d = float(norm_stats['target_2d']['mean'])
                std_2d = max(float(norm_stats['target_2d']['std']), 1e-8)
                baseline_1d = (baseline_1d_physical - mean_1d) / std_1d
                baseline_2d = (baseline_2d_physical - mean_2d) / std_2d
            else:
                baseline_1d = baseline_1d_physical
                baseline_2d = baseline_2d_physical

            model.decoder_1d.set_baseline_from_data(baseline_1d.to(device))
            model.decoder_2d.set_baseline_from_data(baseline_2d.to(device))
            print(f"    1D baseline range (physical): [{baseline_1d_physical.min():.2f}, {baseline_1d_physical.max():.2f}]")
            print(f"    2D baseline range (physical): [{baseline_2d_physical.min():.2f}, {baseline_2d_physical.max():.2f}]")
            if norm_stats is not None:
                print(f"    1D baseline range (normalized): [{baseline_1d.min():.2f}, {baseline_1d.max():.2f}]")
                print(f"    2D baseline range (normalized): [{baseline_2d.min():.2f}, {baseline_2d.max():.2f}]")

        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {num_params:,}")

        # Setup curriculum scheduler if enabled
        curriculum_scheduler = None
        if args.use_curriculum:
            stages = [int(x) for x in args.curriculum_stages.split(',')]
            curriculum_scheduler = CurriculumScheduler(
                stages=stages,
                epochs_per_stage=args.epochs_per_stage,
                patience_per_stage=args.patience_per_stage,
            )
            print(f"  Curriculum stages: {stages}")

        trainer_module = VGSSMTrainer(
            model=model,
            graph=graph,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            beta_ce=args.beta_ce,
            beta_z=args.beta_z,
            horizon_weighting=args.horizon_weighting,
            horizon_weight_by_valid_count=args.horizon_weight_by_valid_count,
            recon_balance_mode=args.recon_balance_mode,
            recon_weight_1d=args.recon_weight_1d,
            recon_weight_2d=args.recon_weight_2d,
            warmup_epochs=args.warmup_epochs,
            max_epochs=args.max_epochs,
            norm_stats=norm_stats,
            loss_type=args.loss_type,
            use_physics_loss=args.use_physics_loss,
            physics_local_weight=args.physics_local_weight,
            physics_global_weight=args.physics_global_weight,
            dt_seconds=args.dt_seconds,
            physics_loss_mode=args.physics_loss_mode,
            physics_residual_huber_delta=args.physics_residual_huber_delta,
            use_curriculum=args.use_curriculum,
            curriculum_scheduler=curriculum_scheduler,
            future_inlet_mode_train=args.future_inlet_mode_train,
            future_inlet_keep_prob=args.future_inlet_keep_prob,
            use_boundary_loss=not args.no_boundary_loss,
            boundary_loss_weight=args.boundary_loss_weight,
            boundary_delta=args.boundary_delta,
            output_bounds_1d=output_bounds_1d,
            output_bounds_2d=output_bounds_2d,
        )

        checkpoint_dir = Path('checkpoints') / f'model_{args.model_id}' / args.exp_name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        callbacks = [
            ModelCheckpoint(
                dirpath=str(checkpoint_dir),
                filename='{epoch:02d}-{val_std_rmse:.4f}',
                monitor='val/std_rmse',
                mode='min',
                save_top_k=3,
                save_last=True,
            ),
            CurriculumAwareEarlyStopping(
                monitor='val/std_rmse',
                patience=args.patience,
                mode='min',
                verbose=True,
            ),
            LearningRateMonitor(logging_interval='epoch'),
        ]

        if args.use_curriculum and curriculum_scheduler is not None:
            callbacks.append(CurriculumCallback(curriculum_scheduler))

        logger = TensorBoardLogger(save_dir='logs', name=f'model_{args.model_id}', version=args.exp_name)

        print("\n[4/4] Starting training...")
        trainer = pl.Trainer(
            max_epochs=args.max_epochs,
            accelerator=args.accelerator,
            devices=args.devices,
            callbacks=callbacks,
            logger=logger,
            precision=args.precision,
            gradient_clip_val=1.0,
            accumulate_grad_batches=args.accumulate_grad_batches,
            limit_train_batches=args.limit_train_batches,
            limit_val_batches=args.limit_val_batches,
            num_sanity_val_steps=args.num_sanity_val_steps,
            check_val_every_n_epoch=args.check_val_every_n_epoch,
            log_every_n_steps=10,
            enable_progress_bar=True,
        )

        ckpt_path = None
        if args.checkpoint:
            if os.path.exists(args.checkpoint):
                ckpt_path = args.checkpoint
                print(f"Resuming from checkpoint: {ckpt_path}")
            else:
                print(f"Warning: checkpoint not found, starting from scratch: {args.checkpoint}")

        if ckpt_path is None and args.init_ckpt:
            if os.path.exists(args.init_ckpt):
                print(f"Initializing model weights from: {args.init_ckpt}")
                init_checkpoint = torch.load(args.init_ckpt, map_location=device)
                if isinstance(init_checkpoint, dict) and 'state_dict' in init_checkpoint:
                    init_state = strip_model_prefix_from_state_dict(init_checkpoint['state_dict'])
                elif isinstance(init_checkpoint, dict):
                    init_state = strip_model_prefix_from_state_dict(init_checkpoint)
                else:
                    raise RuntimeError(f"Unsupported init checkpoint format: {args.init_ckpt}")

                load_result = model.load_state_dict(init_state, strict=False)
                if load_result.missing_keys:
                    print(f"  Init load warning: missing keys={len(load_result.missing_keys)}")
                if load_result.unexpected_keys:
                    print(f"  Init load warning: unexpected keys={len(load_result.unexpected_keys)}")
            else:
                print(f"Warning: init checkpoint not found, starting from scratch: {args.init_ckpt}")
        elif ckpt_path is not None and args.init_ckpt:
            print("Init checkpoint ignored because resume checkpoint is active.")

        trainer.fit(trainer_module, data_module, ckpt_path=ckpt_path)

        # Save best checkpoint
        import shutil
        best_ckpt = checkpoint_dir / 'best.ckpt'
        if trainer.checkpoint_callback.best_model_path:
            shutil.copy(trainer.checkpoint_callback.best_model_path, best_ckpt)
            print(f"\nBest checkpoint saved to: {best_ckpt}")

        print("\n" + "=" * 70)
        print("Training completed!")
        print(f"Best val/std_rmse: {trainer.callback_metrics.get('val/std_rmse', 'N/A')}")
        print("=" * 70)


# =============================================================================
# PREDICTION / INFERENCE
# =============================================================================

class TestEventDataset(Dataset):
    """Dataset for test event prediction."""

    def __init__(self, data_dir, model_id, event_id, graph, norm_stats, prefix_len=10):
        self.data_dir = data_dir
        self.model_id = model_id
        self.event_id = event_id
        self.graph = graph
        self.norm_stats = norm_stats
        self.prefix_len = prefix_len

        self._load_data()

    def _load_data(self):
        event_path = os.path.join(self.data_dir, f"Model_{self.model_id}", "test", f"event_{self.event_id}")

        df_1d = pd.read_csv(os.path.join(event_path, "1d_nodes_dynamic_all.csv"))
        df_2d = pd.read_csv(os.path.join(event_path, "2d_nodes_dynamic_all.csv"))
        df_ts = pd.read_csv(os.path.join(event_path, "timesteps.csv"))

        self.timesteps = df_ts['timestep_idx'].values
        self.num_timesteps = len(self.timesteps)
        self.num_1d_nodes = df_1d['node_idx'].nunique()
        self.num_2d_nodes = df_2d['node_idx'].nunique()

        # Get node IDs for submission
        self.node_ids_1d = sorted(df_1d['node_idx'].unique())
        self.node_ids_2d = sorted(df_2d['node_idx'].unique())

        # Load dynamic data
        input_vars_1d = ['water_level', 'inlet_flow']
        input_vars_2d = ['rainfall', 'water_level', 'water_volume']

        self.dynamic_1d = self._reshape_dynamic(df_1d, self.num_1d_nodes, input_vars_1d)
        self.dynamic_2d = self._reshape_dynamic(df_2d, self.num_2d_nodes, input_vars_2d)

        # For test events, only prefix has water level
        self.prefix_1d = self.dynamic_1d[:self.prefix_len].copy()
        self.prefix_2d = self.dynamic_2d[:self.prefix_len].copy()

        # Future rainfall for all timesteps after prefix
        self.future_rainfall = self.dynamic_2d[self.prefix_len:, :, 0:1].copy()

        # Initial water levels for delta mode
        self.h0_1d = self.dynamic_1d[self.prefix_len - 1:self.prefix_len, :, 0:1].copy()
        self.h0_2d = self.dynamic_2d[self.prefix_len - 1:self.prefix_len, :, 1:2].copy()

    def _reshape_dynamic(self, df, num_nodes, vars):
        available_vars = [v for v in vars if v in df.columns]
        if not available_vars:
            return np.zeros((self.num_timesteps, num_nodes, 1), dtype=np.float32)

        df = df.sort_values(['timestep', 'node_idx'])
        num_timesteps = df['timestep'].nunique()
        data = df[available_vars].values.astype(np.float32)
        data = data.reshape(num_timesteps, num_nodes, len(available_vars))

        # Handle NaN values
        data = np.nan_to_num(data, nan=0.0)

        return data

    def get_normalized_data(self):
        """Get normalized prefix and future rainfall."""
        prefix_1d = self.prefix_1d.copy()
        prefix_2d = self.prefix_2d.copy()
        future_rainfall = self.future_rainfall.copy()
        h0_1d = self.h0_1d.copy()
        h0_2d = self.h0_2d.copy()

        if self.norm_stats is not None:
            prefix_1d = (prefix_1d - self.norm_stats['1d']['mean']) / self.norm_stats['1d']['std']
            prefix_2d = (prefix_2d - self.norm_stats['2d']['mean']) / self.norm_stats['2d']['std']
            future_rainfall = (future_rainfall - self.norm_stats['2d']['mean'][0]) / self.norm_stats['2d']['std'][0]
            h0_1d = (h0_1d - self.norm_stats['target_1d']['mean']) / self.norm_stats['target_1d']['std']
            h0_2d = (h0_2d - self.norm_stats['target_2d']['mean']) / self.norm_stats['target_2d']['std']

        return {
            'prefix_1d': torch.from_numpy(prefix_1d).unsqueeze(0),
            'prefix_2d': torch.from_numpy(prefix_2d).unsqueeze(0),
            'future_rainfall': torch.from_numpy(future_rainfall).unsqueeze(0),
            'h0_1d': torch.from_numpy(h0_1d).unsqueeze(0),
            'h0_2d': torch.from_numpy(h0_2d).unsqueeze(0),
        }


def calibrate_latents(model, graph, data, warmup_targets_1d, warmup_targets_2d,
                      norm_stats, calibration_steps=50, lr=0.01, device='cuda'):
    """Calibrate c_e, z0_1d, z0_2d using warmup period."""
    model.eval()

    prefix_1d = data['prefix_1d'].to(device)
    prefix_2d = data['prefix_2d'].to(device)
    future_rainfall = data['future_rainfall'].to(device)

    # Initialize latents from encoder
    with torch.no_grad():
        spatial_1d, spatial_2d = model.encode_spatial(graph)
        c_e, c_e_mean, _ = model.encode_event_latent(prefix_1d, prefix_2d, deterministic=True)
        z0_mu_1d, _ = model.z0_encoder_1d(prefix_1d, spatial_1d, c_e)
        z0_mu_2d, _ = model.z0_encoder_2d(prefix_2d, spatial_2d, c_e)

    # Create optimizable parameters
    c_e_opt = nn.Parameter(c_e_mean.clone())
    z0_1d_opt = nn.Parameter(z0_mu_1d.clone())
    z0_2d_opt = nn.Parameter(z0_mu_2d.clone())

    optimizer = torch.optim.Adam([c_e_opt, z0_1d_opt, z0_2d_opt], lr=lr)

    warmup_len = warmup_targets_1d.shape[1]

    for step in range(calibration_steps):
        optimizer.zero_grad()

        outputs = model(
            graph, prefix_1d, prefix_2d,
            prefix_len=prefix_1d.shape[1],
            future_rainfall=future_rainfall[:, :warmup_len],
            c_e_override=c_e_opt,
            z0_1d_override=z0_1d_opt,
            z0_2d_override=z0_2d_opt,
            deterministic_latent=True,
        )

        pred_1d = outputs['pred_1d'][:, :warmup_len, :, 0]
        pred_2d = outputs['pred_2d'][:, :warmup_len, :, 0]

        loss = F.mse_loss(pred_1d, warmup_targets_1d) + F.mse_loss(pred_2d, warmup_targets_2d)
        loss.backward()
        optimizer.step()

    return c_e_opt.detach(), z0_1d_opt.detach(), z0_2d_opt.detach()


def predict_event(model, graph, data, norm_stats, device='cuda',
                  calibrate=False, calibration_steps=50, deterministic_latent=True):
    """Generate predictions for a single test event."""
    model.eval()

    prefix_1d = data['prefix_1d'].to(device)
    prefix_2d = data['prefix_2d'].to(device)
    future_rainfall = data['future_rainfall'].to(device)
    h0_1d = data.get('h0_1d')
    h0_2d = data.get('h0_2d')

    if h0_1d is not None:
        h0_1d = h0_1d.to(device).squeeze(1)  # [batch, num_nodes, 1]
    if h0_2d is not None:
        h0_2d = h0_2d.to(device).squeeze(1)

    with torch.no_grad():
        if calibrate:
            c_e_override = None
            z0_1d_override = None
            z0_2d_override = None
        else:
            c_e_override = None
            z0_1d_override = None
            z0_2d_override = None

        outputs = model(
            graph, prefix_1d, prefix_2d,
            prefix_len=prefix_1d.shape[1],
            future_rainfall=future_rainfall,
            c_e_override=c_e_override,
            z0_1d_override=z0_1d_override,
            z0_2d_override=z0_2d_override,
            h0_1d=h0_1d,
            h0_2d=h0_2d,
            deterministic_latent=deterministic_latent,
        )

        pred_1d = outputs['pred_1d'][0, :, :, 0].cpu().numpy()
        pred_2d = outputs['pred_2d'][0, :, :, 0].cpu().numpy()

    # Denormalize
    if norm_stats is not None:
        pred_1d = pred_1d * norm_stats['target_1d']['std'] + norm_stats['target_1d']['mean']
        pred_2d = pred_2d * norm_stats['target_2d']['std'] + norm_stats['target_2d']['mean']

    return pred_1d, pred_2d


def predict_event_autoregressive(
    model,
    graph,
    data,
    norm_stats,
    device='cuda',
    max_timesteps=400,
    chunk_size=90,
    deterministic_latent=True,
    stateful_rollout: bool = False,
    c_e_override=None,
    z0_1d_override=None,
    z0_2d_override=None,
):
    """Generate predictions using autoregressive rollout for long sequences."""
    model.eval()

    prefix_1d = data['prefix_1d'].to(device)
    prefix_2d = data['prefix_2d'].to(device)
    future_rainfall = data['future_rainfall'].to(device)

    h0_1d = data.get('h0_1d')
    h0_2d = data.get('h0_2d')

    if h0_1d is not None:
        h0_1d = h0_1d.to(device).squeeze(1)
    if h0_2d is not None:
        h0_2d = h0_2d.to(device).squeeze(1)

    all_pred_1d = []
    all_pred_2d = []
    current_c_e_override = c_e_override
    current_z0_1d_override = z0_1d_override
    current_z0_2d_override = z0_2d_override

    total_rainfall_steps = future_rainfall.shape[1]
    steps_predicted = 0

    with torch.no_grad():
        while steps_predicted < max_timesteps and steps_predicted < total_rainfall_steps:
            rain_start = steps_predicted
            rain_end = min(steps_predicted + chunk_size, total_rainfall_steps)
            chunk_rainfall = future_rainfall[:, rain_start:rain_end, :, :]  # [batch, chunk, num_2d, 1]

            # Pad rainfall if needed
            actual_chunk = chunk_rainfall.shape[1]
            if actual_chunk < chunk_size:
                pad_size = chunk_size - actual_chunk
                chunk_rainfall = torch.cat([
                    chunk_rainfall,
                    chunk_rainfall[:, -1:, :, :].expand(-1, pad_size, -1, -1)
                ], dim=1)

            outputs = model(
                graph, prefix_1d, prefix_2d,
                prefix_len=prefix_1d.shape[1],
                future_rainfall=chunk_rainfall,
                c_e_override=current_c_e_override,
                z0_1d_override=current_z0_1d_override,
                z0_2d_override=current_z0_2d_override,
                h0_1d=h0_1d,
                h0_2d=h0_2d,
                deterministic_latent=deterministic_latent,
                return_final_state=True,
            )

            pred_1d = outputs['pred_1d'][0]  # [T, N, 1]
            pred_2d = outputs['pred_2d'][0]

            steps_to_use = min(chunk_size, rain_end - rain_start, max_timesteps - steps_predicted)

            all_pred_1d.append(pred_1d[:steps_to_use, :, 0].cpu())
            all_pred_2d.append(pred_2d[:steps_to_use, :, 0].cpu())

            steps_predicted += steps_to_use

            if steps_predicted >= max_timesteps or steps_predicted >= total_rainfall_steps:
                del outputs, pred_1d, pred_2d
                torch.cuda.empty_cache()
                break

            if stateful_rollout:
                # Carry latent/event state across chunks.
                current_c_e_override = outputs['c_e']
                current_z0_1d_override = outputs['zT_1d']
                current_z0_2d_override = outputs['zT_2d']
            else:
                # Reset latent state per chunk; keep any explicit event override and hydraulic continuity via h0.
                current_c_e_override = c_e_override
                current_z0_1d_override = z0_1d_override
                current_z0_2d_override = z0_2d_override

            # Update h0 for next chunk with last predicted (normalized) values
            if 'hT_1d' in outputs and 'hT_2d' in outputs:
                h0_1d = outputs['hT_1d']
                h0_2d = outputs['hT_2d']
            else:
                h0_1d = pred_1d[-1, :, :].unsqueeze(0).clone()  # [1, N, 1]
                h0_2d = pred_2d[-1, :, :].unsqueeze(0).clone()

            # Clear memory
            del outputs, pred_1d, pred_2d
            torch.cuda.empty_cache()

    pred_1d = torch.cat(all_pred_1d, dim=0).numpy()
    pred_2d = torch.cat(all_pred_2d, dim=0).numpy()

    # Denormalize
    if norm_stats is not None:
        pred_1d = pred_1d * norm_stats['target_1d']['std'] + norm_stats['target_1d']['mean']
        pred_2d = pred_2d * norm_stats['target_2d']['std'] + norm_stats['target_2d']['mean']

    return pred_1d, pred_2d


def _get_checkpoint_best_monitor_score(
    checkpoint: dict,
    monitor_candidates: Tuple[str, ...] = ('val/std_rmse', 'val_std_rmse'),
) -> Tuple[Optional[str], Optional[float]]:
    """Extract best monitored validation metric from Lightning checkpoint callbacks."""
    callbacks = checkpoint.get('callbacks', {})
    for cb_state in callbacks.values():
        if not isinstance(cb_state, dict):
            continue
        monitor = cb_state.get('monitor')
        if monitor not in monitor_candidates:
            continue
        best_score = cb_state.get('best_model_score')
        if best_score is None:
            return monitor, None
        try:
            return monitor, float(best_score)
        except (TypeError, ValueError):
            try:
                return monitor, float(best_score.item())
            except Exception:
                return monitor, None
    return None, None


def _validate_checkpoint_quality_for_prediction(checkpoint: dict, args) -> None:
    """Fail fast when trying to generate submissions from low-quality checkpoints."""
    monitor, best_score = _get_checkpoint_best_monitor_score(checkpoint)
    if monitor is None:
        print("  Warning: no checkpoint monitor metadata found; skipping checkpoint quality gate")
        return

    print(f"  Checkpoint best {monitor}: {best_score if best_score is not None else 'unknown'}")
    threshold = float(getattr(args, 'predict_max_ckpt_std_rmse', 0.0))
    if threshold > 0 and best_score is not None and best_score > threshold:
        raise RuntimeError(
            f"Checkpoint quality gate failed: best {monitor}={best_score:.4f} > "
            f"allowed {threshold:.4f}. Use a better checkpoint or increase --predict_max_ckpt_std_rmse."
        )


def _summarize_and_validate_prediction_distribution(submission_df: pd.DataFrame, norm_stats: dict, args) -> None:
    """Report and validate output distribution against train-target statistics."""
    summary = (
        submission_df
        .groupby('node_type')['water_level']
        .agg(['count', 'mean', 'std', 'min', 'max'])
        .reindex(['1d', '2d'])
    )
    print("\nPrediction distribution summary:")
    print(summary)

    max_std_ratio = float(getattr(args, 'predict_max_std_ratio', 0.0))
    if max_std_ratio <= 0 or norm_stats is None:
        return

    for node_type, target_key in (('1d', 'target_1d'), ('2d', 'target_2d')):
        if node_type not in summary.index:
            continue
        pred_std = summary.loc[node_type, 'std']
        if pd.isna(pred_std):
            continue
        train_std = float(norm_stats[target_key]['std'])
        ratio = float(pred_std) / max(train_std, 1e-8)
        print(
            f"  node_type={node_type}: pred_std={float(pred_std):.4f}, "
            f"train_std={train_std:.4f}, ratio={ratio:.3f}"
        )
        if ratio > max_std_ratio:
            raise RuntimeError(
                f"Prediction distribution gate failed for {node_type}: "
                f"std ratio {ratio:.3f} > allowed {max_std_ratio:.3f}. "
                "Likely undertrained/wrong checkpoint or unstable rollout."
            )


def generate_submission(args):
    """Generate submission file for a model."""
    print(f"\n{'='*70}")
    print(f"Generating predictions for Model {args.model_id}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"{'='*70}")

    device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load checkpoint
    print("\n[1/4] Loading checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    _validate_checkpoint_quality_for_prediction(checkpoint, args)

    # Get hyperparameters from checkpoint
    hparams = checkpoint.get('hyper_parameters', {})

    # Build graph
    print("\n[2/4] Building graph...")
    builder = FloodGraphBuilder(args.data_dir, args.model_id, add_knn_2d_edges=True, knn_k=8)
    graph = builder.build(split="test")
    graph = graph.to(device)

    static_1d_dim = graph['1d'].x.shape[1]
    static_2d_dim = graph['2d'].x.shape[1]
    print(f"  1D nodes: {graph['1d'].x.shape[0]}")
    print(f"  2D nodes: {graph['2d'].x.shape[0]}")

    # Get default model config (may be overridden by checkpoint inference below).
    model_config = get_model_specific_config(args.model_id, args)

    # Compute normalization stats from the exact train split used in training.
    # This must match FloodDataModule.setup() to avoid train/inference scale mismatch.
    print("\n[3/4] Computing normalization stats (training split)...")
    train_events, val_events = split_train_val_events(args.data_dir, args.model_id, val_ratio=0.2, seed=42)
    print(f"  Train events for stats: {len(train_events)} (val events excluded: {len(val_events)})")
    norm_stats = compute_norm_stats_for_events(
        data_dir=args.data_dir,
        model_id=args.model_id,
        event_ids=train_events,
        graph=graph,
        seq_len=args.seq_len,
        pred_len=args.prediction_horizon,
        stride=args.stride,
    )

    # Create model
    print("\n[4/4] Creating model...")
    (
        output_bounds_1d_physical,
        output_bounds_2d_physical,
        output_bounds_1d,
        output_bounds_2d,
    ) = resolve_internal_output_bounds(args.model_id, args, norm_stats)
    print(f"  Using output bounds 1D (physical): {output_bounds_1d_physical}")
    print(f"  Using output bounds 2D (physical): {output_bounds_2d_physical}")
    print(f"  Using output bounds 1D (normalized): {output_bounds_1d}")
    print(f"  Using output bounds 2D (normalized): {output_bounds_2d}")

    use_timer = getattr(args, 'use_timer', False)
    use_timer_v4 = getattr(args, 'use_timer_v4', False)
    timer_v4_pooling = getattr(args, 'timer_v4_pooling', 'mean')
    use_grassmann = getattr(args, 'use_grassmann', False)
    grassmann_offsets = None
    if hasattr(args, 'grassmann_offsets') and args.grassmann_offsets:
        grassmann_offsets = [int(x) for x in args.grassmann_offsets.split(',')]

    state_dict = strip_model_prefix_from_state_dict(checkpoint['state_dict'])
    inferred_config = infer_model_config_from_state_dict(
        state_dict=state_dict,
        fallback_model_config=model_config,
        fallback_event_latent_dim=args.event_latent_dim,
        fallback_num_heads=args.num_heads,
    )
    print(
        "  Inferred model config: "
        f"hidden_dim={inferred_config['hidden_dim']}, "
        f"latent_dim={inferred_config['latent_dim']}, "
        f"event_latent_dim={inferred_config['event_latent_dim']}, "
        f"gnn_layers={inferred_config['num_gnn_layers']}, "
        f"transition_gnn_layers={inferred_config['num_transition_gnn_layers']}, "
        f"num_heads={inferred_config['num_heads']}"
    )

    model = VGSSM(
        static_1d_dim=static_1d_dim,
        static_2d_dim=static_2d_dim,
        dynamic_1d_dim=2,
        dynamic_2d_dim=3,
        hidden_dim=inferred_config['hidden_dim'],
        latent_dim=inferred_config['latent_dim'],
        event_latent_dim=inferred_config['event_latent_dim'],
        num_gnn_layers=inferred_config['num_gnn_layers'],
        num_transition_gnn_layers=inferred_config['num_transition_gnn_layers'],
        num_heads=inferred_config['num_heads'],
        prediction_horizon=args.prediction_horizon,
        use_event_latent=True,
        dropout=inferred_config['dropout'],
        use_delta_prediction=args.use_delta_prediction,
        use_physics_loss=args.use_physics_loss,
        physics_subsample_rate=getattr(args, 'physics_subsample_rate', 5),
        # Timer settings
        use_timer=use_timer,
        timer_layers=getattr(args, 'timer_layers', 4),
        timer_heads=getattr(args, 'timer_heads', 4),
        timer_history_len=getattr(args, 'timer_history_len', 10),
        # Timer V4 settings (scientific fix)
        use_timer_v4=use_timer_v4,
        timer_v4_pooling=timer_v4_pooling,
        timer_transition_variant=getattr(args, 'timer_transition_variant', 'auto'),
        timer_enable_2d_context=bool(getattr(args, 'timer_enable_2d_context', False)),
        # Grassmann Flow settings (attention-free)
        use_grassmann=use_grassmann,
        grassmann_layers=getattr(args, 'grassmann_layers', 4),
        grassmann_rank=getattr(args, 'grassmann_rank', 12),
        grassmann_offsets=grassmann_offsets,
        # Physics-constrained transition
        use_physics_transition=getattr(args, 'use_physics_transition', False),
        # Output bounds to prevent invalid predictions
        output_bounds_1d=output_bounds_1d,
        output_bounds_2d=output_bounds_2d,
        # Sigmoid bounds: h = min + sigmoid(x) * (max - min)
        use_sigmoid_bounds=getattr(args, 'use_sigmoid_bounds', False),
        # Physics decoder: K -> Q -> V -> h (mass-conserving)
        use_physics_decoder=getattr(args, 'use_physics_decoder', False),
        num_1d_nodes=graph['1d'].x.shape[0],
        num_2d_nodes=graph['2d'].x.shape[0],
        physics_dt=getattr(args, 'physics_dt', 300.0),
        latent_sample_temperature=args.latent_sample_temperature,
        latent_state_clip=args.latent_state_clip,
    )

    # Load state dict
    load_result = model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys:
        print(f"  Warning: missing keys while loading checkpoint: {len(load_result.missing_keys)}")
    if load_result.unexpected_keys:
        print(f"  Warning: unexpected keys while loading checkpoint: {len(load_result.unexpected_keys)}")
    model = model.to(device)
    model.eval()
    print(f"  Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters")
    deterministic_latent = not bool(getattr(args, 'predict_stochastic_latent', False))
    print(f"  Latent inference mode: {'deterministic (posterior mean)' if deterministic_latent else 'stochastic sampling'}")

    # Discover test events
    test_path = os.path.join(args.data_dir, f"Model_{args.model_id}", "test")
    test_events = []
    for item in os.listdir(test_path):
        if item.startswith("event_"):
            try:
                test_events.append(int(item.split("_")[1]))
            except ValueError:
                pass
    test_events = sorted(test_events)
    print(f"  Found {len(test_events)} test events")

    # Generate predictions
    print("\nGenerating predictions...")
    all_rows = []

    for event_id in test_events:
        print(f"  Event {event_id}...", end=" ", flush=True)

        # Load test event
        test_ds = TestEventDataset(args.data_dir, args.model_id, event_id, graph, norm_stats, prefix_len=10)
        data = test_ds.get_normalized_data()

        # Predict with autoregressive rollout for long sequences
        prefix_len = 10
        pred_timesteps = test_ds.timesteps[prefix_len:]
        needed_timesteps = len(pred_timesteps)

        if args.predict_chunk_size > 0:
            chunk_size = int(args.predict_chunk_size)
        else:
            chunk_size = max(1, min(int(args.prediction_horizon), int(needed_timesteps)))

        pred_1d, pred_2d = predict_event_autoregressive(
            model, graph, data, norm_stats, device=device,
            max_timesteps=needed_timesteps, chunk_size=chunk_size,
            deterministic_latent=deterministic_latent,
            stateful_rollout=bool(getattr(args, 'predict_rollout_stateful', False)),
        )

        # Apply post-hoc clamping to ensure valid water level range
        # Model 1: observed range [286, 362], bounds [280, 370]
        # Model 2: observed range [22, 60], bounds [15, 70]
        wl_min_1d, wl_max_1d = output_bounds_1d_physical
        wl_min_2d, wl_max_2d = output_bounds_2d_physical
        pred_1d = np.clip(pred_1d, wl_min_1d, wl_max_1d)
        pred_2d = np.clip(pred_2d, wl_min_2d, wl_max_2d)

        # Create submission rows
        # 1D predictions
        for t_idx, timestep in enumerate(pred_timesteps):
            if t_idx >= pred_1d.shape[0]:
                # Fallback: pad with last value if still short
                val_1d = pred_1d[-1, :]
            else:
                val_1d = pred_1d[t_idx, :]
            for n_idx, node_id in enumerate(test_ds.node_ids_1d):
                all_rows.append({
                    'model_id': args.model_id,
                    'event_id': event_id,
                    'node_type': '1d',
                    'node_idx': node_id,
                    'timestep': timestep,
                    'water_level': float(val_1d[n_idx]),
                })

        # 2D predictions
        for t_idx, timestep in enumerate(pred_timesteps):
            if t_idx >= pred_2d.shape[0]:
                val_2d = pred_2d[-1, :]
            else:
                val_2d = pred_2d[t_idx, :]
            for n_idx, node_id in enumerate(test_ds.node_ids_2d):
                all_rows.append({
                    'model_id': args.model_id,
                    'event_id': event_id,
                    'node_type': '2d',
                    'node_idx': node_id,
                    'timestep': timestep,
                    'water_level': float(val_2d[n_idx]),
                })

        print(f"done ({pred_1d.shape[0]}/{needed_timesteps} steps predicted)")

    # Create submission DataFrame
    print("\nCreating submission file...")
    submission_df = pd.DataFrame(all_rows)
    _summarize_and_validate_prediction_distribution(submission_df, norm_stats, args)

    # Save
    output_file = f"submission_vgssm_physics_model{args.model_id}.parquet"
    submission_df.to_parquet(output_file, index=False)
    print(f"Saved to {output_file}")
    print(f"  Total rows: {len(submission_df):,}")
    print(f"  Water level range: [{submission_df['water_level'].min():.2f}, {submission_df['water_level'].max():.2f}]")

    return output_file


def compute_per_node_baseline(data_dir: str, model_id: int, graph) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-node mean water level from training data.

    This is used for baseline residual mode where the model predicts
    deviations from the per-node mean rather than absolute values.

    Returns:
        baseline_1d: Mean water level per 1D node [num_1d_nodes]
        baseline_2d: Mean water level per 2D node [num_2d_nodes]
    """
    import os

    train_path = os.path.join(data_dir, f"Model_{model_id}", "train")
    events = sorted([d for d in os.listdir(train_path) if d.startswith('event_')])

    num_1d = graph['1d'].num_nodes
    num_2d = graph['2d'].num_nodes

    # Accumulate sum and count per node
    sum_1d = torch.zeros(num_1d)
    sum_2d = torch.zeros(num_2d)
    count_1d = torch.zeros(num_1d)
    count_2d = torch.zeros(num_2d)

    for event_dir in events:
        event_path = os.path.join(train_path, event_dir)

        # Load 1D data
        df_1d = pd.read_csv(os.path.join(event_path, "1d_nodes_dynamic_all.csv"))
        for node_idx in range(num_1d):
            node_data = df_1d[df_1d['node_idx'] == node_idx]['water_level']
            if len(node_data) > 0:
                sum_1d[node_idx] += node_data.sum()
                count_1d[node_idx] += len(node_data)

        # Load 2D data
        df_2d = pd.read_csv(os.path.join(event_path, "2d_nodes_dynamic_all.csv"))
        for node_idx in range(num_2d):
            node_data = df_2d[df_2d['node_idx'] == node_idx]['water_level']
            if len(node_data) > 0:
                sum_2d[node_idx] += node_data.sum()
                count_2d[node_idx] += len(node_data)

    # Compute mean (avoid division by zero)
    baseline_1d = sum_1d / count_1d.clamp(min=1)
    baseline_2d = sum_2d / count_2d.clamp(min=1)

    return baseline_1d, baseline_2d


if __name__ == '__main__':
    main()
