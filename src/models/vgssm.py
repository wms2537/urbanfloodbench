"""
Variational Graph State-Space Model (VGSSM) for Urban Flood Prediction.

Upgrades Graph-TFT with per-timestep latent dynamics (z_t) in addition to
event latent (c_e). This models hidden hydraulic states that evolve over time.

Architecture:
1. Spatial Encoder: HeteroGNN for static graph structure (reused from Graph-TFT)
2. Event Encoder: Encodes c_e from prefix observations (reused)
3. Latent Inference Net: Infers z_0 from prefix given c_e
4. Latent Transition: z_{t+1} = z_t + f(z_t, graph, u_t, c_e)
5. Latent Decoder: y_t = g(z_t, spatial_emb)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv
from typing import Dict, Optional, Tuple

from .coupled_gnn import SpatialEncoder
from .graph_tft import EventLatentEncoderTFT
from .tft import GatedResidualNetwork


class HeteroGNNBlock(nn.Module):
    """
    Lightweight heterogeneous GNN block for latent state transitions.
    Propagates information across 1D-2D coupling.
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # Define edge types for 1D-2D coupling
        self.edge_types = [
            ('1d', 'pipe', '1d'),
            ('2d', 'surface', '2d'),
            ('1d', 'couples_to', '2d'),
            ('2d', 'couples_from', '1d'),
        ]

        # Separate convolutions for each edge type
        conv_dict = {}
        for edge_type in self.edge_types:
            conv_dict[edge_type] = SAGEConv(
                (latent_dim, latent_dim),
                hidden_dim,
                aggr='mean',
            )
        self.conv = HeteroConv(conv_dict, aggr='sum')

        # Output projections
        self.proj_1d = nn.Linear(hidden_dim, latent_dim)
        self.proj_2d = nn.Linear(hidden_dim, latent_dim)

        self.norm_1d = nn.LayerNorm(latent_dim)
        self.norm_2d = nn.LayerNorm(latent_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        z_1d: torch.Tensor,
        z_2d: torch.Tensor,
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply graph message passing to latent states.

        Args:
            z_1d: [batch, num_1d_nodes, latent_dim]
            z_2d: [batch, num_2d_nodes, latent_dim]
            edge_index_dict: Edge indices for all edge types

        Returns:
            delta_z_1d: [batch, num_1d_nodes, latent_dim]
            delta_z_2d: [batch, num_2d_nodes, latent_dim]
        """
        batch_size = z_1d.shape[0]
        num_1d = z_1d.shape[1]
        num_2d = z_2d.shape[1]

        # Process each batch element
        out_1d_list = []
        out_2d_list = []

        for b in range(batch_size):
            x_dict = {
                '1d': z_1d[b],  # [num_1d, latent_dim]
                '2d': z_2d[b],  # [num_2d, latent_dim]
            }

            # Apply heterogeneous convolution
            out_dict = self.conv(x_dict, edge_index_dict)

            # Handle missing outputs (some node types might not have incoming edges)
            out_1d = out_dict.get('1d', torch.zeros(num_1d, self.proj_1d.out_features, device=z_1d.device))
            out_2d = out_dict.get('2d', torch.zeros(num_2d, self.proj_2d.out_features, device=z_2d.device))

            out_1d_list.append(out_1d)
            out_2d_list.append(out_2d)

        out_1d = torch.stack(out_1d_list, dim=0)  # [batch, num_1d, hidden_dim]
        out_2d = torch.stack(out_2d_list, dim=0)  # [batch, num_2d, hidden_dim]

        # Project back to latent dim
        delta_1d = self.proj_1d(out_1d)
        delta_2d = self.proj_2d(out_2d)

        # Normalize and dropout
        delta_1d = self.dropout(self.norm_1d(delta_1d))
        delta_2d = self.dropout(self.norm_2d(delta_2d))

        return delta_1d, delta_2d


class LatentTransition(nn.Module):
    """
    Latent state transition model: z_{t+1} = z_t + f(z_t, graph, u_t, c_e)

    Combines:
    1. Spatial propagation via GNN (information flows through graph)
    2. Temporal update via MLP (external forcing from rainfall/inflow)
    3. Event conditioning via c_e (event-specific behavior)
    """

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        c_e_dim: int,
        control_dim_1d: int,  # inlet_flow dimension
        control_dim_2d: int,  # rainfall dimension
        num_gnn_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # GNN blocks for spatial propagation
        self.gnn_blocks = nn.ModuleList([
            HeteroGNNBlock(latent_dim, hidden_dim, dropout)
            for _ in range(num_gnn_layers)
        ])

        # MLP for temporal update (1D nodes)
        self.temporal_mlp_1d = nn.Sequential(
            nn.Linear(latent_dim + c_e_dim + control_dim_1d, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

        # MLP for temporal update (2D nodes) - includes rainfall
        self.temporal_mlp_2d = nn.Sequential(
            nn.Linear(latent_dim + c_e_dim + control_dim_2d, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Gating for residual update
        self.gate_1d = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Sigmoid(),
        )
        self.gate_2d = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        z_1d: torch.Tensor,
        z_2d: torch.Tensor,
        edge_index_dict: Dict[Tuple[str, str, str], torch.Tensor],
        u_1d: Optional[torch.Tensor],  # [batch, num_1d_nodes, control_dim_1d]
        u_2d: torch.Tensor,  # [batch, num_2d_nodes, control_dim_2d] (rainfall)
        c_e: torch.Tensor,  # [batch, c_e_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute next latent state.

        Returns:
            z_1d_next: [batch, num_1d_nodes, latent_dim]
            z_2d_next: [batch, num_2d_nodes, latent_dim]
        """
        batch, num_1d, _ = z_1d.shape
        num_2d = z_2d.shape[1]

        # 1. Spatial propagation through GNN
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

        # 2. Temporal update with controls
        c_e_1d = c_e.unsqueeze(1).expand(-1, num_1d, -1)  # [batch, num_1d, c_e_dim]
        c_e_2d = c_e.unsqueeze(1).expand(-1, num_2d, -1)  # [batch, num_2d, c_e_dim]

        # 1D temporal update
        if u_1d is not None:
            input_1d = torch.cat([z_1d, c_e_1d, u_1d], dim=-1)
        else:
            input_1d = torch.cat([z_1d, c_e_1d], dim=-1)
            # Pad with zeros if no control
            zeros = torch.zeros(batch, num_1d, self.temporal_mlp_1d[0].in_features - input_1d.shape[-1], device=z_1d.device)
            input_1d = torch.cat([input_1d, zeros], dim=-1)

        delta_temporal_1d = self.temporal_mlp_1d(input_1d)

        # 2D temporal update (always has rainfall)
        input_2d = torch.cat([z_2d, c_e_2d, u_2d], dim=-1)
        delta_temporal_2d = self.temporal_mlp_2d(input_2d)

        # 3. Combine spatial and temporal with gating
        delta_1d = delta_spatial_1d + delta_temporal_1d
        delta_2d = delta_spatial_2d + delta_temporal_2d

        # Gated update
        gate_1d = self.gate_1d(torch.cat([z_1d, delta_1d], dim=-1))
        gate_2d = self.gate_2d(torch.cat([z_2d, delta_2d], dim=-1))

        z_1d_next = z_1d + gate_1d * delta_1d
        z_2d_next = z_2d + gate_2d * delta_2d

        return z_1d_next, z_2d_next


class LatentInferenceNet(nn.Module):
    """
    Inference network for initial latent state: q(z_0 | prefix, c_e)

    Uses a GRU to encode prefix observations and outputs (mu, logvar) for z_0.
    """

    def __init__(
        self,
        input_dim: int,  # dynamic features per node
        hidden_dim: int,
        latent_dim: int,
        c_e_dim: int,
        spatial_dim: int,  # spatial embedding dimension
        num_nodes: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_nodes = num_nodes

        # Input projection
        self.input_proj = nn.Linear(input_dim + spatial_dim + c_e_dim, hidden_dim)

        # GRU for temporal encoding (shared across nodes)
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout,
        )

        # Per-node output projections
        self.mu_proj = nn.Linear(hidden_dim, latent_dim)
        self.logvar_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        prefix_seq: torch.Tensor,  # [batch, prefix_len, num_nodes, input_dim]
        spatial_emb: torch.Tensor,  # [num_nodes, spatial_dim]
        c_e: torch.Tensor,  # [batch, c_e_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode prefix into initial latent distribution.

        Returns:
            mu: [batch, num_nodes, latent_dim]
            logvar: [batch, num_nodes, latent_dim]
        """
        batch, prefix_len, num_nodes, input_dim = prefix_seq.shape

        # Expand spatial embeddings and c_e
        spatial_expanded = spatial_emb.unsqueeze(0).unsqueeze(0).expand(batch, prefix_len, -1, -1)
        c_e_expanded = c_e.unsqueeze(1).unsqueeze(2).expand(-1, prefix_len, num_nodes, -1)

        # Concatenate inputs
        combined = torch.cat([prefix_seq, spatial_expanded, c_e_expanded], dim=-1)

        # Project
        h = self.input_proj(combined)  # [batch, prefix_len, num_nodes, hidden_dim]

        # Reshape for per-node GRU: [batch * num_nodes, prefix_len, hidden_dim]
        h = h.permute(0, 2, 1, 3).reshape(batch * num_nodes, prefix_len, -1)

        # GRU encoding
        _, h_final = self.gru(h)  # h_final: [num_layers, batch*num_nodes, hidden_dim]
        h_final = h_final[-1]  # Take last layer: [batch*num_nodes, hidden_dim]

        # Project to latent parameters
        mu = self.mu_proj(h_final)  # [batch*num_nodes, latent_dim]
        logvar = self.logvar_proj(h_final)
        logvar = torch.clamp(logvar, min=-10, max=2)  # Prevent extreme values

        # Reshape: [batch, num_nodes, latent_dim]
        mu = mu.view(batch, num_nodes, self.latent_dim)
        logvar = logvar.view(batch, num_nodes, self.latent_dim)

        return mu, logvar


class LatentDecoder(nn.Module):
    """
    Decodes latent state to water level predictions: y_t = g(z_t, spatial_emb)
    """

    def __init__(
        self,
        latent_dim: int,
        spatial_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + spatial_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(
        self,
        z: torch.Tensor,  # [batch, num_nodes, latent_dim]
        spatial_emb: torch.Tensor,  # [num_nodes, spatial_dim]
    ) -> torch.Tensor:
        """
        Decode latent state to predictions.

        Returns:
            pred: [batch, num_nodes, output_dim]
        """
        batch = z.shape[0]
        spatial_expanded = spatial_emb.unsqueeze(0).expand(batch, -1, -1)
        combined = torch.cat([z, spatial_expanded], dim=-1)
        return self.decoder(combined)


def reparameterize(mu: torch.Tensor, logvar: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """Reparameterization trick for sampling from N(mu, sigma^2)."""
    std = torch.exp(0.5 * logvar) * temperature
    eps = torch.randn_like(std)
    return mu + eps * std


class VGSSM(nn.Module):
    """
    Variational Graph State-Space Model for Urban Flood Prediction.

    Combines:
    - Event latent c_e: captures event-level characteristics
    - Per-timestep latent z_t: models evolving hydraulic state
    - Graph structure: handles 1D-2D coupling
    """

    def __init__(
        self,
        # Node feature dimensions
        static_1d_dim: int,
        static_2d_dim: int,
        dynamic_1d_dim: int,
        dynamic_2d_dim: int,
        # Architecture
        hidden_dim: int = 64,
        latent_dim: int = 32,
        event_latent_dim: int = 16,
        num_gnn_layers: int = 3,
        num_transition_gnn_layers: int = 2,
        num_heads: int = 4,
        # Multi-horizon prediction
        prediction_horizon: int = 90,
        # Options
        use_event_latent: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.event_latent_dim = event_latent_dim if use_event_latent else 0
        self.use_event_latent = use_event_latent
        self.prediction_horizon = prediction_horizon
        self.dynamic_1d_dim = dynamic_1d_dim
        self.dynamic_2d_dim = dynamic_2d_dim

        # 1. Spatial encoder (heterogeneous GNN for static features)
        self.spatial_encoder = SpatialEncoder(
            static_1d_dim, static_2d_dim,
            hidden_channels=hidden_dim,
            num_layers=num_gnn_layers,
            use_attention=True,
            dropout=dropout,
        )

        # 2. Event latent encoder
        if use_event_latent:
            self.event_encoder = EventLatentEncoderTFT(
                input_dim=dynamic_1d_dim + dynamic_2d_dim,
                hidden_dim=hidden_dim,
                latent_dim=event_latent_dim,
                num_heads=num_heads,
                dropout=dropout,
            )
            # Prior parameters
            self.event_prior_mean = nn.Parameter(torch.zeros(event_latent_dim))
            self.event_prior_logvar = nn.Parameter(torch.zeros(event_latent_dim))

        # 3. Latent inference networks for z_0
        self.z0_encoder_1d = LatentInferenceNet(
            input_dim=dynamic_1d_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            c_e_dim=self.event_latent_dim,
            spatial_dim=hidden_dim,
            num_nodes=1,  # Will be determined at runtime
            dropout=dropout,
        )
        self.z0_encoder_2d = LatentInferenceNet(
            input_dim=dynamic_2d_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            c_e_dim=self.event_latent_dim,
            spatial_dim=hidden_dim,
            num_nodes=1,  # Will be determined at runtime
            dropout=dropout,
        )

        # Prior for z_0
        self.z0_prior_mean = nn.Parameter(torch.zeros(latent_dim))
        self.z0_prior_logvar = nn.Parameter(torch.zeros(latent_dim))

        # 4. Latent transition models
        control_dim_1d = 1  # inlet_flow (or 0 if not available)
        control_dim_2d = 1  # rainfall

        self.transition = LatentTransition(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            c_e_dim=self.event_latent_dim,
            control_dim_1d=control_dim_1d,
            control_dim_2d=control_dim_2d,
            num_gnn_layers=num_transition_gnn_layers,
            dropout=dropout,
        )

        # 5. Latent decoders
        self.output_dim_1d = 1  # water_level
        self.output_dim_2d = 1  # water_level

        self.decoder_1d = LatentDecoder(
            latent_dim=latent_dim,
            spatial_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=self.output_dim_1d,
            dropout=dropout,
        )
        self.decoder_2d = LatentDecoder(
            latent_dim=latent_dim,
            spatial_dim=hidden_dim,
            hidden_dim=hidden_dim,
            output_dim=self.output_dim_2d,
            dropout=dropout,
        )

    def encode_spatial(self, graph: HeteroData) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode static spatial features using heterogeneous GNN."""
        return self.spatial_encoder(graph)

    def encode_event_latent(
        self,
        prefix_1d: torch.Tensor,
        prefix_2d: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode event latent from prefix observations.

        Args:
            prefix_1d: [batch, prefix_len, num_1d_nodes, dynamic_1d_dim]
            prefix_2d: [batch, prefix_len, num_2d_nodes, dynamic_2d_dim]

        Returns:
            c_e: Sampled event latent [batch, event_latent_dim]
            mean: [batch, event_latent_dim]
            logvar: [batch, event_latent_dim]
        """
        if not self.use_event_latent:
            batch = prefix_1d.shape[0]
            device = prefix_1d.device
            zeros = torch.zeros(batch, 0, device=device)
            return zeros, zeros, zeros

        # Pool features across nodes
        pooled_1d = prefix_1d.mean(dim=2)  # [batch, prefix_len, dynamic_1d_dim]
        pooled_2d = prefix_2d.mean(dim=2)  # [batch, prefix_len, dynamic_2d_dim]

        # Combine as synthetic "1 node" representation
        combined = torch.cat([pooled_1d, pooled_2d], dim=-1).unsqueeze(2)

        mean, logvar = self.event_encoder(combined)
        c_e = self.event_encoder.sample(mean, logvar)

        return c_e, mean, logvar

    def forward(
        self,
        graph: HeteroData,
        input_1d: torch.Tensor,
        input_2d: torch.Tensor,
        prefix_len: int = 10,
        future_rainfall: Optional[torch.Tensor] = None,
        future_inlet_flow: Optional[torch.Tensor] = None,
        c_e_override: Optional[torch.Tensor] = None,
        z0_1d_override: Optional[torch.Tensor] = None,
        z0_2d_override: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with latent state rollout.

        Args:
            graph: Static graph structure
            input_1d: [batch, seq_len, num_1d_nodes, dynamic_1d_dim]
            input_2d: [batch, seq_len, num_2d_nodes, dynamic_2d_dim]
            prefix_len: Length of prefix for encoding
            future_rainfall: [batch, horizon, num_2d_nodes, 1] - known future rainfall
            future_inlet_flow: [batch, horizon, num_1d_nodes, 1] - optional future inlet flow
            c_e_override: Optional pre-computed event latent
            z0_1d_override: Optional pre-computed initial latent for 1D
            z0_2d_override: Optional pre-computed initial latent for 2D

        Returns:
            Dict with predictions and latent distributions
        """
        batch, seq_len, num_1d_nodes, _ = input_1d.shape
        num_2d_nodes = input_2d.shape[2]
        device = input_1d.device

        # 1. Encode spatial features (static)
        spatial_1d, spatial_2d = self.encode_spatial(graph)

        # 2. Extract edge indices for transition GNN
        edge_index_dict = {}
        for edge_type in self.transition.gnn_blocks[0].edge_types:
            if edge_type in graph.edge_types:
                edge_index_dict[edge_type] = graph[edge_type].edge_index

        # 3. Encode event latent from prefix
        if c_e_override is not None:
            c_e = c_e_override
            c_e_mean = c_e_override
            c_e_logvar = torch.zeros_like(c_e_override)
        else:
            prefix_1d = input_1d[:, :prefix_len]
            prefix_2d = input_2d[:, :prefix_len]
            c_e, c_e_mean, c_e_logvar = self.encode_event_latent(prefix_1d, prefix_2d)

        # 4. Infer initial latent state z_0
        if z0_1d_override is not None:
            z0_mu_1d = z0_1d_override
            z0_logvar_1d = torch.zeros_like(z0_1d_override)
            z_1d = z0_1d_override
        else:
            prefix_1d = input_1d[:, :prefix_len]
            z0_mu_1d, z0_logvar_1d = self.z0_encoder_1d(prefix_1d, spatial_1d, c_e)
            z_1d = reparameterize(z0_mu_1d, z0_logvar_1d)

        if z0_2d_override is not None:
            z0_mu_2d = z0_2d_override
            z0_logvar_2d = torch.zeros_like(z0_2d_override)
            z_2d = z0_2d_override
        else:
            prefix_2d = input_2d[:, :prefix_len]
            z0_mu_2d, z0_logvar_2d = self.z0_encoder_2d(prefix_2d, spatial_2d, c_e)
            z_2d = reparameterize(z0_mu_2d, z0_logvar_2d)

        # 5. Rollout latent dynamics
        horizon = min(self.prediction_horizon, future_rainfall.shape[1]) if future_rainfall is not None else self.prediction_horizon

        preds_1d = []
        preds_2d = []
        z_1d_trajectory = [z_1d]
        z_2d_trajectory = [z_2d]

        for t in range(horizon):
            # Get controls at timestep t
            u_2d = future_rainfall[:, t] if future_rainfall is not None else torch.zeros(batch, num_2d_nodes, 1, device=device)
            u_1d = future_inlet_flow[:, t] if future_inlet_flow is not None else None

            # Transition to next state
            z_1d, z_2d = self.transition(
                z_1d, z_2d, edge_index_dict, u_1d, u_2d, c_e
            )

            # Decode current state to predictions
            pred_1d = self.decoder_1d(z_1d, spatial_1d)  # [batch, num_1d, output_dim]
            pred_2d = self.decoder_2d(z_2d, spatial_2d)  # [batch, num_2d, output_dim]

            preds_1d.append(pred_1d)
            preds_2d.append(pred_2d)
            z_1d_trajectory.append(z_1d)
            z_2d_trajectory.append(z_2d)

        # Stack predictions: [batch, horizon, num_nodes, output_dim]
        pred_1d = torch.stack(preds_1d, dim=1)
        pred_2d = torch.stack(preds_2d, dim=1)

        return {
            'pred_1d': pred_1d,
            'pred_2d': pred_2d,
            'c_e': c_e,
            'c_e_mean': c_e_mean,
            'c_e_logvar': c_e_logvar,
            'z0_mu_1d': z0_mu_1d,
            'z0_logvar_1d': z0_logvar_1d,
            'z0_mu_2d': z0_mu_2d,
            'z0_logvar_2d': z0_logvar_2d,
            'z_1d_trajectory': torch.stack(z_1d_trajectory, dim=1),
            'z_2d_trajectory': torch.stack(z_2d_trajectory, dim=1),
        }

    def forward_from_latents(
        self,
        graph: HeteroData,
        c_e: torch.Tensor,
        z0_1d: torch.Tensor,
        z0_2d: torch.Tensor,
        future_rainfall: torch.Tensor,
        future_inlet_flow: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass from pre-specified latent states (for test-time calibration).

        Args:
            graph: Static graph
            c_e: Event latent [batch, event_latent_dim]
            z0_1d: Initial latent for 1D [batch, num_1d, latent_dim]
            z0_2d: Initial latent for 2D [batch, num_2d, latent_dim]
            future_rainfall: [batch, horizon, num_2d_nodes, 1]
            future_inlet_flow: Optional [batch, horizon, num_1d_nodes, 1]

        Returns:
            Dict with predictions
        """
        batch = c_e.shape[0]
        num_1d_nodes = z0_1d.shape[1]
        num_2d_nodes = z0_2d.shape[1]
        device = c_e.device

        # Get spatial embeddings
        spatial_1d, spatial_2d = self.encode_spatial(graph)

        # Edge indices
        edge_index_dict = {}
        for edge_type in self.transition.gnn_blocks[0].edge_types:
            if edge_type in graph.edge_types:
                edge_index_dict[edge_type] = graph[edge_type].edge_index

        # Rollout from provided initial states
        horizon = future_rainfall.shape[1]
        z_1d = z0_1d
        z_2d = z0_2d

        preds_1d = []
        preds_2d = []

        for t in range(horizon):
            u_2d = future_rainfall[:, t]
            u_1d = future_inlet_flow[:, t] if future_inlet_flow is not None else None

            z_1d, z_2d = self.transition(
                z_1d, z_2d, edge_index_dict, u_1d, u_2d, c_e
            )

            pred_1d = self.decoder_1d(z_1d, spatial_1d)
            pred_2d = self.decoder_2d(z_2d, spatial_2d)

            preds_1d.append(pred_1d)
            preds_2d.append(pred_2d)

        return {
            'pred_1d': torch.stack(preds_1d, dim=1),
            'pred_2d': torch.stack(preds_2d, dim=1),
        }

    def optimize_latents(
        self,
        graph: HeteroData,
        prefix_1d: torch.Tensor,
        prefix_2d: torch.Tensor,
        warmup_target_1d: torch.Tensor,
        warmup_target_2d: torch.Tensor,
        warmup_rainfall: torch.Tensor,
        num_steps: int = 50,
        lr: float = 0.01,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Test-time optimization of latent states (c_e, z0_1d, z0_2d).

        Uses warmup period targets to calibrate latents before prediction.

        Args:
            graph: Static graph
            prefix_1d: [batch, prefix_len, num_1d, dynamic_1d_dim]
            prefix_2d: [batch, prefix_len, num_2d, dynamic_2d_dim]
            warmup_target_1d: [batch, warmup_len, num_1d] water levels
            warmup_target_2d: [batch, warmup_len, num_2d] water levels
            warmup_rainfall: [batch, warmup_len, num_2d, 1]
            num_steps: Optimization steps
            lr: Learning rate

        Returns:
            Optimized (c_e, z0_1d, z0_2d)
        """
        # Initialize from posterior
        with torch.no_grad():
            spatial_1d, spatial_2d = self.encode_spatial(graph)
            c_e, _, _ = self.encode_event_latent(prefix_1d, prefix_2d)
            z0_mu_1d, _ = self.z0_encoder_1d(prefix_1d, spatial_1d, c_e)
            z0_mu_2d, _ = self.z0_encoder_2d(prefix_2d, spatial_2d, c_e)

        # Make optimizable
        c_e = c_e.clone().detach().requires_grad_(True)
        z0_1d = z0_mu_1d.clone().detach().requires_grad_(True)
        z0_2d = z0_mu_2d.clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([c_e, z0_1d, z0_2d], lr=lr)

        warmup_len = warmup_target_1d.shape[1]

        for step in range(num_steps):
            optimizer.zero_grad()

            # Forward with current latents
            outputs = self.forward_from_latents(
                graph, c_e, z0_1d, z0_2d,
                warmup_rainfall[:, :warmup_len],
            )

            # Loss on warmup period
            pred_1d = outputs['pred_1d'][:, :warmup_len, :, 0]
            pred_2d = outputs['pred_2d'][:, :warmup_len, :, 0]

            loss_1d = F.mse_loss(pred_1d, warmup_target_1d)
            loss_2d = F.mse_loss(pred_2d, warmup_target_2d)

            # Regularization toward prior
            kl_ce = 0.01 * ((c_e - self.event_prior_mean) ** 2).mean()
            kl_z0 = 0.001 * (z0_1d ** 2).mean() + 0.001 * (z0_2d ** 2).mean()

            loss = loss_1d + loss_2d + kl_ce + kl_z0
            loss.backward()
            optimizer.step()

        return c_e.detach(), z0_1d.detach(), z0_2d.detach()
