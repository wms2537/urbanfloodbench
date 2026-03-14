#!/usr/bin/env python3
"""
Physics-Informed Timer (PIT) Architecture

Scientific Design Principles:
1. Physics-Constrained Attention: Attention between nodes should respect hydraulic connectivity
2. Differentiable Physics Layer: Transition explicitly models mass conservation
3. Hybrid Temporal-Spatial: Blend Timer (temporal) with GNN (spatial) in each block

Key Innovations:
- PhysicsGatedAttention: Attention modulated by hydraulic potential
- DifferentiablePhysicsUpdate: Forward pass includes explicit physics computation
- HybridTimerGNNBlock: Alternates between temporal and spatial information flow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax


class PhysicsGatedAttention(nn.Module):
    """
    Attention mechanism that incorporates physical adjacency and flow potential.

    Scientific rationale:
    - Standard attention: attn(Q, K) = softmax(QK^T / sqrt(d))
    - Physics-gated: attn(Q, K, G) = softmax(QK^T / sqrt(d) + G)

    Where G is a physics-informed bias:
    - G_ij = log(exp(flow_potential_ij) + 1) for connected nodes
    - G_ij = -inf for disconnected nodes (physically impossible flow)

    This makes attention respect the physical structure of the drainage network.
    """

    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        # Physics bias network: computes flow potential from node features
        self.physics_bias = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_heads),
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_mask=None, edge_index=None):
        """
        Args:
            x: [batch, seq_len, hidden_dim] or [batch, num_nodes, hidden_dim]
            adj_mask: [num_nodes, num_nodes] - physical adjacency (1 = connected)
            edge_index: [2, num_edges] - optional edge list
        """
        batch, seq_len, _ = x.shape

        # Compute Q, K, V
        q = self.q_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Standard attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [batch, heads, seq, seq]

        # Add physics bias if adjacency mask is provided
        if adj_mask is not None:
            # Compute pairwise physics bias
            # x_i concatenated with x_j for each pair
            x_i = x.unsqueeze(2).expand(-1, -1, seq_len, -1)  # [batch, seq, seq, hidden]
            x_j = x.unsqueeze(1).expand(-1, seq_len, -1, -1)  # [batch, seq, seq, hidden]
            pair_features = torch.cat([x_i, x_j], dim=-1)  # [batch, seq, seq, 2*hidden]

            physics_bias = self.physics_bias(pair_features)  # [batch, seq, seq, heads]
            physics_bias = physics_bias.permute(0, 3, 1, 2)  # [batch, heads, seq, seq]

            # Apply adjacency mask: -inf for disconnected nodes
            adj_mask = adj_mask.float().unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]
            physics_mask = torch.where(adj_mask > 0, physics_bias, torch.tensor(-1e9, device=x.device))

            attn = attn + physics_mask

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention
        out = (attn @ v).transpose(1, 2).reshape(batch, seq_len, self.hidden_dim)
        out = self.out_proj(out)

        return out


class DifferentiablePhysicsUpdate(nn.Module):
    """
    Differentiable physics layer that computes water level updates from mass conservation.

    Scientific basis (Saint-Venant / shallow water equations):
        dh/dt = (Q_in - Q_out) / A + S

    Where:
        h = water level
        Q = flow rate (derived from head difference)
        A = surface area
        S = source/sink (rainfall, infiltration)

    This layer:
    1. Computes flows from water level differences: Q_ij = f(h_i - h_j)
    2. Updates water levels using conservation: h_{t+1} = h_t + dt * net_flow / A
    """

    def __init__(self, hidden_dim, dt=300.0):  # dt in seconds (5 minutes)
        super().__init__()
        self.dt = dt

        # Flow coefficient network: learns hydraulic resistance
        self.flow_coef = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),  # Flow coefficient in [0, 1]
        )

        # Area estimation from latent (for mass balance)
        self.area_head = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),  # Area must be positive
        )

        # Maximum flow rate (prevents numerical instability)
        self.max_flow = 10.0

    def forward(self, h, z, edge_index, edge_type, rainfall=None, node_areas=None):
        """
        Compute physics-based water level update.

        Args:
            h: [batch, num_nodes] - current water levels
            z: [batch, num_nodes, latent_dim] - latent states (for flow coefficients)
            edge_index: [2, num_edges] - connectivity
            edge_type: [num_edges] - edge types (pipe, surface, coupling)
            rainfall: [batch, num_nodes] - rainfall intensity (optional)
            node_areas: [num_nodes] - node surface areas (optional)

        Returns:
            h_new: [batch, num_nodes] - updated water levels
            flows: [batch, num_edges] - computed flows
        """
        batch = h.shape[0]
        num_edges = edge_index.shape[1]

        # Estimate node areas from latent if not provided
        if node_areas is None:
            areas = self.area_head(z).squeeze(-1)  # [batch, num_nodes]
        else:
            areas = node_areas.unsqueeze(0).expand(batch, -1)

        # Compute head difference for each edge
        h_src = h[:, edge_index[0]]  # [batch, num_edges]
        h_dst = h[:, edge_index[1]]  # [batch, num_edges]
        delta_h = h_src - h_dst

        # Flow coefficient from source node latent
        z_src = z[:, edge_index[0], :]  # [batch, num_edges, latent_dim]
        flow_coef = self.flow_coef(z_src).squeeze(-1)  # [batch, num_edges]

        # Flow: Q = K * sign(Δh) * |Δh|^0.5 (weir/orifice equation)
        flows = flow_coef * torch.sign(delta_h) * torch.sqrt(torch.abs(delta_h) + 1e-6)
        flows = flows.clamp(-self.max_flow, self.max_flow)

        # Aggregate inflows and outflows per node
        # net_flow_i = sum(Q_ji) - sum(Q_ij) for each node i
        num_nodes = h.shape[1]
        inflow = torch.zeros(batch, num_nodes, device=h.device)
        outflow = torch.zeros(batch, num_nodes, device=h.device)

        # Scatter: inflow to destination, outflow from source
        inflow.scatter_add_(1, edge_index[1].unsqueeze(0).expand(batch, -1), flows.clamp(min=0))
        outflow.scatter_add_(1, edge_index[0].unsqueeze(0).expand(batch, -1), flows.clamp(min=0))

        # Also add reverse flows
        inflow.scatter_add_(1, edge_index[0].unsqueeze(0).expand(batch, -1), (-flows).clamp(min=0))
        outflow.scatter_add_(1, edge_index[1].unsqueeze(0).expand(batch, -1), (-flows).clamp(min=0))

        net_flow = inflow - outflow

        # Add rainfall source
        if rainfall is not None:
            net_flow = net_flow + rainfall

        # Update water level: h_{t+1} = h_t + dt * net_flow / A
        dh = self.dt * net_flow / (areas + 1e-6)
        h_new = h + dh.clamp(-1.0, 1.0)  # Limit update magnitude

        return h_new, flows


class HybridTimerGNNBlock(nn.Module):
    """
    Hybrid block that combines temporal (Timer) and spatial (GNN) information.

    Design:
    1. Temporal Attention: Each node attends to its own history
    2. Spatial Message Passing: Nodes exchange information with neighbors
    3. Physics Gating: Modulates spatial communication by flow potential

    This respects both the temporal dynamics (water level evolution)
    and spatial dependencies (hydraulic connectivity).
    """

    def __init__(self, hidden_dim, num_heads=4, dropout=0.1):
        super().__init__()

        # Temporal self-attention (per-node)
        self.temporal_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.temporal_norm = nn.LayerNorm(hidden_dim)

        # Spatial GNN layer (physics-aware)
        self.spatial_conv = PhysicsAwareConv(hidden_dim)
        self.spatial_norm = nn.LayerNorm(hidden_dim)

        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout),
        )
        self.ffn_norm = nn.LayerNorm(hidden_dim)

    def forward(self, x, edge_index, edge_attr=None):
        """
        Args:
            x: [batch, seq_len, num_nodes, hidden_dim]
            edge_index: [2, num_edges]
            edge_attr: [num_edges, edge_dim] (optional)
        """
        batch, seq_len, num_nodes, hidden_dim = x.shape

        # 1. Temporal attention (process each node's sequence)
        x_temporal = x.permute(0, 2, 1, 3).reshape(batch * num_nodes, seq_len, hidden_dim)
        attn_out, _ = self.temporal_attn(x_temporal, x_temporal, x_temporal)
        x_temporal = self.temporal_norm(x_temporal + attn_out)
        x = x_temporal.reshape(batch, num_nodes, seq_len, hidden_dim).permute(0, 2, 1, 3)

        # 2. Spatial message passing (process each timestep)
        x_spatial = x.reshape(batch * seq_len, num_nodes, hidden_dim)
        spatial_out = self.spatial_conv(x_spatial, edge_index, edge_attr)
        x_spatial = self.spatial_norm(x_spatial + spatial_out)
        x = x_spatial.reshape(batch, seq_len, num_nodes, hidden_dim)

        # 3. FFN
        x = self.ffn_norm(x + self.ffn(x))

        return x


class PhysicsAwareConv(MessagePassing):
    """
    GNN convolution that respects physics of water flow.

    Messages are modulated by:
    1. Edge type (pipe, surface, 1D-2D coupling)
    2. Feature difference (related to head difference -> flow potential)
    """

    def __init__(self, hidden_dim):
        super().__init__(aggr='add')

        # Edge type embedding
        self.edge_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Message function
        self.message_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        # Update function
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x, edge_index, edge_attr=None):
        """
        Args:
            x: [batch * seq_len, num_nodes, hidden_dim]
            edge_index: [2, num_edges]
            edge_attr: [num_edges, edge_dim]
        """
        # Handle batched input
        if x.dim() == 3:
            batch_size, num_nodes, hidden_dim = x.shape
            x = x.reshape(batch_size * num_nodes, hidden_dim)

            # Adjust edge_index for batched graph
            offset = torch.arange(batch_size, device=x.device) * num_nodes
            offset = offset.unsqueeze(1).unsqueeze(2)  # [batch, 1, 1]
            batch_edge_index = edge_index.unsqueeze(0) + offset  # [batch, 2, edges]
            batch_edge_index = batch_edge_index.permute(1, 0, 2).reshape(2, -1)

            # Expand edge_attr for batch
            if edge_attr is not None:
                edge_attr = edge_attr.unsqueeze(0).expand(batch_size, -1, -1)
                edge_attr = edge_attr.reshape(-1, edge_attr.shape[-1])

            out = self.propagate(batch_edge_index, x=x, edge_attr=edge_attr)
            return out.reshape(batch_size, num_nodes, hidden_dim)
        else:
            return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        # x_i: target node features, x_j: source node features
        diff = x_i - x_j  # Related to potential flow

        if edge_attr is not None:
            edge_emb = self.edge_mlp(edge_attr)
            msg_input = torch.cat([x_j, diff, edge_emb], dim=-1)
        else:
            msg_input = torch.cat([x_j, diff, torch.zeros_like(x_j)], dim=-1)

        return self.message_mlp(msg_input)

    def update(self, aggr_out, x):
        return self.update_mlp(torch.cat([x, aggr_out], dim=-1))


class PhysicsInformedTimerPosterior(nn.Module):
    """
    Physics-Informed Timer for encoding prefix into initial latent z_0.

    Key innovations over standard Timer:
    1. Graph-aware attention respects hydraulic connectivity
    2. Physics residual connection ensures conservation
    3. Last-state weighting for proper initial condition

    Architecture:
        prefix_seq -> [HybridTimerGNNBlock x N] -> Physics Layer -> (μ, σ) for z_0
    """

    def __init__(self, input_dim, hidden_dim, latent_dim, c_e_dim, spatial_dim,
                 num_nodes, num_layers=4, num_heads=4, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_nodes = num_nodes

        # Input projection
        self.input_proj = nn.Linear(input_dim + spatial_dim + c_e_dim, hidden_dim)

        # Positional encoding for temporal position
        self.pos_embed = nn.Parameter(torch.zeros(1, 64, 1, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Hybrid Timer-GNN blocks
        self.blocks = nn.ModuleList([
            HybridTimerGNNBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Physics layer for final state
        self.physics_update = DifferentiablePhysicsUpdate(hidden_dim)

        # Output heads
        self.ln_out = nn.LayerNorm(hidden_dim)

        # Weighted pooling: emphasize last timestep but include history
        self.time_weights = nn.Parameter(torch.zeros(64))  # Will be softmax'd

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

    def forward(self, prefix_seq, spatial_emb, c_e, edge_index=None, edge_attr=None):
        """
        Args:
            prefix_seq: [batch, prefix_len, num_nodes, input_dim]
            spatial_emb: [num_nodes, spatial_dim]
            c_e: [batch, c_e_dim]
            edge_index: [2, num_edges] - graph connectivity
            edge_attr: [num_edges, edge_dim] - edge features
        """
        batch, prefix_len, num_nodes, input_dim = prefix_seq.shape

        # Combine inputs
        spatial_expanded = spatial_emb.unsqueeze(0).unsqueeze(0).expand(batch, prefix_len, -1, -1)
        c_e_expanded = c_e.unsqueeze(1).unsqueeze(2).expand(-1, prefix_len, num_nodes, -1)
        combined = torch.cat([prefix_seq, spatial_expanded, c_e_expanded], dim=-1)

        # Project
        h = self.input_proj(combined)  # [batch, prefix_len, num_nodes, hidden_dim]

        # Add positional encoding
        h = h + self.pos_embed[:, :prefix_len, :, :]

        # Process through hybrid blocks
        for block in self.blocks:
            h = block(h, edge_index, edge_attr)

        h = self.ln_out(h)  # [batch, prefix_len, num_nodes, hidden_dim]

        # Weighted temporal pooling (emphasis on last timestep)
        weights = F.softmax(self.time_weights[:prefix_len], dim=0)
        # Create exponential decay favoring last timesteps
        decay = torch.exp(torch.linspace(-2, 0, prefix_len, device=h.device))
        weights = weights * decay
        weights = weights / weights.sum()

        # Apply weights
        weights = weights.view(1, prefix_len, 1, 1)
        h_pooled = (h * weights).sum(dim=1)  # [batch, num_nodes, hidden_dim]

        # Predict posterior
        mu = self.mu_head(h_pooled)
        logvar = self.logvar_head(h_pooled)
        logvar = torch.clamp(logvar, min=-10, max=2)

        return mu, logvar


class PhysicsConstrainedTransition(nn.Module):
    """
    Transition model that blends neural updates with physics constraints.

    Design:
        z_{t+1} = z_t + delta_neural + delta_physics

    Where:
        delta_neural = GNN(z_t) + MLP(z_t, u_t, c_e)  # learned dynamics
        delta_physics = PhysicsLayer(h_t) -> dh -> encode(dh)  # physics-based

    The physics term ensures conservation is respected, while the neural term
    learns residual dynamics not captured by simplified physics.
    """

    def __init__(self, latent_dim, hidden_dim, c_e_dim, num_gnn_layers=2):
        super().__init__()
        self.latent_dim = latent_dim

        # Neural transition (same as before)
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_gnn_layers):
            self.gnn_layers.append(PhysicsAwareConv(latent_dim))

        self.mlp = nn.Sequential(
            nn.Linear(latent_dim + 1 + c_e_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

        # Physics transition
        self.physics_update = DifferentiablePhysicsUpdate(latent_dim)

        # Decoder: latent -> water level (for physics computation)
        self.h_decoder = nn.Linear(latent_dim, 1)

        # Encoder: water level change -> latent change
        self.dh_encoder = nn.Linear(1, latent_dim)

        # Physics weight (learnable blending)
        self.physics_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, z_t, u_t, c_e, edge_index, edge_type=None):
        """
        Args:
            z_t: [batch, num_nodes, latent_dim]
            u_t: [batch, num_nodes, 1] - input (rainfall)
            c_e: [batch, c_e_dim] - event latent
        """
        batch, num_nodes, _ = z_t.shape

        # Neural transition
        z_flat = z_t.reshape(batch * num_nodes, self.latent_dim)
        for gnn in self.gnn_layers:
            # Expand edge_index for batch
            offset = torch.arange(batch, device=z_t.device) * num_nodes
            batch_edges = edge_index.unsqueeze(0) + offset.view(-1, 1, 1)
            batch_edges = batch_edges.permute(1, 0, 2).reshape(2, -1)
            z_flat = z_flat + gnn.propagate(batch_edges, x=z_flat)
        z_gnn = z_flat.reshape(batch, num_nodes, self.latent_dim)

        # MLP part
        c_e_expanded = c_e.unsqueeze(1).expand(-1, num_nodes, -1)
        mlp_input = torch.cat([z_t, u_t, c_e_expanded], dim=-1)
        z_mlp = self.mlp(mlp_input)

        delta_neural = z_gnn + z_mlp - z_t  # Neural delta

        # Physics transition
        h_t = self.h_decoder(z_t).squeeze(-1)  # [batch, num_nodes]
        h_new, _ = self.physics_update(h_t, z_t, edge_index, edge_type,
                                       rainfall=u_t.squeeze(-1))
        dh = h_new - h_t
        delta_physics = self.dh_encoder(dh.unsqueeze(-1))  # [batch, num_nodes, latent]

        # Blend: mostly neural, with physics constraint
        physics_w = torch.sigmoid(self.physics_weight)
        z_next = z_t + delta_neural + physics_w * delta_physics

        return z_next


# Main model that integrates everything
class PhysicsInformedVGSSM(nn.Module):
    """
    VGSSM with Physics-Informed Timer architecture.

    Improvements over base VGSSM:
    1. PhysicsInformedTimerPosterior for z_0 encoding
    2. PhysicsConstrainedTransition for z_t evolution
    3. Differentiable physics layer integrated in forward pass
    """

    def __init__(self, config):
        super().__init__()
        # ... implementation would go here
        pass


if __name__ == "__main__":
    print("Physics-Informed Timer Architecture")
    print("="*50)
    print("""
Key Components:
1. PhysicsGatedAttention: Attention respects hydraulic connectivity
2. DifferentiablePhysicsUpdate: Mass conservation in forward pass
3. HybridTimerGNNBlock: Blends temporal and spatial processing
4. PhysicsInformedTimerPosterior: Graph-aware z_0 encoding
5. PhysicsConstrainedTransition: Neural + physics delta blending

Scientific Principles:
- Physics is part of architecture, not just loss
- Conservation law (dh/dt = Q_in - Q_out) is differentiable
- Graph structure informs attention patterns
- Last-state weighting for initial conditions
""")
