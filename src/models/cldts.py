"""
CL-DTS: Coupled Latent Digital Twin Surrogate
Main model combining spatial GNN, temporal processing, and variational inference.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from typing import Dict, Optional, Tuple

from .coupled_gnn import CoupledHeteroGNN, SpatialEncoder
from .temporal import TemporalBlock, SpatioTemporalEncoder


class EventLatentEncoder(nn.Module):
    """
    Encodes event-level latent variable c_e from prefix observations.
    c_e captures event-specific unknowns: roughness shifts, blockages, inlet efficiency.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int = 2,
    ):
        """
        Args:
            input_dim: Dimension of input features per node
            hidden_dim: Hidden dimension
            latent_dim: Dimension of event latent c_e
            num_layers: Number of encoder layers
        """
        super().__init__()
        self.latent_dim = latent_dim

        # Temporal aggregator (processes prefix of observations)
        self.temporal = TemporalBlock(input_dim, hidden_dim, num_layers)

        # Global pooling + latent projection
        self.pool_proj = nn.Linear(hidden_dim, hidden_dim)

        # Variational: output mean and log-variance
        self.mean_proj = nn.Linear(hidden_dim, latent_dim)
        self.logvar_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode prefix observations into event latent.

        Args:
            x: Input sequence [batch, prefix_len, num_nodes, input_dim]
            mask: Optional mask for variable-length prefixes

        Returns:
            mean: [batch, latent_dim]
            logvar: [batch, latent_dim]
        """
        batch, seq_len, num_nodes, input_dim = x.shape

        # Process temporal sequence
        output, _ = self.temporal(x)  # [batch, seq_len, num_nodes, hidden_dim]

        # Global pooling: mean over time and nodes
        if mask is not None:
            # Masked mean
            mask = mask.unsqueeze(-1).unsqueeze(-1)  # [batch, seq_len, 1, 1]
            output = output * mask
            pooled = output.sum(dim=(1, 2)) / (mask.sum(dim=1) * num_nodes + 1e-8)
        else:
            pooled = output.mean(dim=(1, 2))  # [batch, hidden_dim]

        # Project
        pooled = F.gelu(self.pool_proj(pooled))

        # Compute mean and log-variance
        mean = self.mean_proj(pooled)
        logvar = self.logvar_proj(pooled)

        # Clamp logvar for stability
        logvar = torch.clamp(logvar, min=-10, max=2)

        return mean, logvar

    def sample(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Reparameterization trick for sampling.

        Args:
            mean: [batch, latent_dim]
            logvar: [batch, latent_dim]
            temperature: Sampling temperature (1.0 = standard)

        Returns:
            sample: [batch, latent_dim]
        """
        std = torch.exp(0.5 * logvar) * temperature
        eps = torch.randn_like(std)
        return mean + eps * std


class DynamicLatentTransition(nn.Module):
    """
    Transition model for dynamic latent z_t.
    z_t captures hidden physical state (flow potential, velocity field).
    p(z_{t+1} | z_t, y_t, u_t, c_e)
    """

    def __init__(
        self,
        latent_dim: int,
        obs_dim: int,
        event_latent_dim: int,
        hidden_dim: int,
    ):
        """
        Args:
            latent_dim: Dimension of dynamic latent z_t
            obs_dim: Dimension of observations y_t
            event_latent_dim: Dimension of event latent c_e
            hidden_dim: Hidden dimension
        """
        super().__init__()
        self.latent_dim = latent_dim

        input_dim = latent_dim + obs_dim + event_latent_dim

        self.transition = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Output mean and log-variance for z_{t+1}
        self.mean_proj = nn.Linear(hidden_dim, latent_dim)
        self.logvar_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        z_t: torch.Tensor,
        y_t: torch.Tensor,
        c_e: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute transition distribution parameters.

        Args:
            z_t: Current latent state [batch, num_nodes, latent_dim]
            y_t: Current observations [batch, num_nodes, obs_dim]
            c_e: Event latent [batch, event_latent_dim]

        Returns:
            mean: [batch, num_nodes, latent_dim]
            logvar: [batch, num_nodes, latent_dim]
        """
        batch, num_nodes, _ = z_t.shape

        # Expand event latent to all nodes
        c_e_expanded = c_e.unsqueeze(1).expand(-1, num_nodes, -1)

        # Concatenate inputs
        inputs = torch.cat([z_t, y_t, c_e_expanded], dim=-1)

        # Forward
        h = self.transition(inputs)
        mean = self.mean_proj(h)
        logvar = self.logvar_proj(h)

        # Clamp for stability
        logvar = torch.clamp(logvar, min=-10, max=2)

        return mean, logvar

    def sample(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std


class InferenceNetwork(nn.Module):
    """
    Inference network q(z_t | y_{<=t}, c_e).
    Acts as a learned filter for data assimilation.
    """

    def __init__(
        self,
        obs_dim: int,
        event_latent_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int = 2,
    ):
        """
        Args:
            obs_dim: Dimension of observations
            event_latent_dim: Dimension of event latent c_e
            hidden_dim: Hidden dimension
            latent_dim: Dimension of dynamic latent z_t
            num_layers: Number of GRU layers
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # Input projection
        self.input_proj = nn.Linear(obs_dim + event_latent_dim, hidden_dim)

        # Recurrent encoder
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers, batch_first=True)

        # Output projections
        self.mean_proj = nn.Linear(hidden_dim, latent_dim)
        self.logvar_proj = nn.Linear(hidden_dim, latent_dim)

    def forward(
        self,
        y_seq: torch.Tensor,
        c_e: torch.Tensor,
        h0: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Infer dynamic latent from observations.

        Args:
            y_seq: Observation sequence [batch, seq_len, num_nodes, obs_dim]
            c_e: Event latent [batch, event_latent_dim]
            h0: Optional initial hidden state

        Returns:
            mean_seq: [batch, seq_len, num_nodes, latent_dim]
            logvar_seq: [batch, seq_len, num_nodes, latent_dim]
            hidden: Final hidden state
        """
        batch, seq_len, num_nodes, obs_dim = y_seq.shape

        # Expand event latent
        c_e_expanded = c_e.unsqueeze(1).unsqueeze(2).expand(-1, seq_len, num_nodes, -1)

        # Concatenate and project
        inputs = torch.cat([y_seq, c_e_expanded], dim=-1)
        inputs = self.input_proj(inputs)

        # Reshape for GRU: [batch * num_nodes, seq_len, hidden_dim]
        inputs = inputs.permute(0, 2, 1, 3).reshape(batch * num_nodes, seq_len, self.hidden_dim)

        # Forward through GRU
        output, hidden = self.gru(inputs, h0)

        # Project to mean and logvar
        mean_seq = self.mean_proj(output)
        logvar_seq = self.logvar_proj(output)

        # Reshape back
        mean_seq = mean_seq.reshape(batch, num_nodes, seq_len, self.latent_dim)
        mean_seq = mean_seq.permute(0, 2, 1, 3)

        logvar_seq = logvar_seq.reshape(batch, num_nodes, seq_len, self.latent_dim)
        logvar_seq = logvar_seq.permute(0, 2, 1, 3)

        # Clamp for stability
        logvar_seq = torch.clamp(logvar_seq, min=-10, max=2)

        return mean_seq, logvar_seq, hidden


class MultiOutputDecoder(nn.Module):
    """
    Decodes multiple output features from latent state and spatial embeddings.
    For true autoregressive prediction, we predict ALL features needed as input.
    """

    def __init__(
        self,
        latent_dim: int,
        spatial_dim: int,
        event_latent_dim: int,
        hidden_dim: int,
        output_dim: int = 1,  # Number of output features
    ):
        super().__init__()
        self.output_dim = output_dim

        input_dim = latent_dim + spatial_dim + event_latent_dim

        self.decoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),  # Multi-output
        )

    def forward(
        self,
        z: torch.Tensor,
        spatial_emb: torch.Tensor,
        c_e: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode multiple output features.

        Args:
            z: Dynamic latent [batch, num_nodes, latent_dim] or [batch, seq_len, num_nodes, latent_dim]
            spatial_emb: Spatial embeddings [num_nodes, spatial_dim]
            c_e: Event latent [batch, event_latent_dim]

        Returns:
            outputs: [batch, num_nodes, output_dim] or [batch, seq_len, num_nodes, output_dim]
        """
        if z.dim() == 3:
            batch, num_nodes, latent_dim = z.shape
            seq_len = None
        else:
            batch, seq_len, num_nodes, latent_dim = z.shape

        # Expand spatial embeddings
        if seq_len is None:
            spatial_expanded = spatial_emb.unsqueeze(0).expand(batch, -1, -1)
            c_e_expanded = c_e.unsqueeze(1).expand(-1, num_nodes, -1)
        else:
            spatial_expanded = spatial_emb.unsqueeze(0).unsqueeze(0).expand(batch, seq_len, -1, -1)
            c_e_expanded = c_e.unsqueeze(1).unsqueeze(2).expand(-1, seq_len, num_nodes, -1)

        # Concatenate
        inputs = torch.cat([z, spatial_expanded, c_e_expanded], dim=-1)

        # Decode - returns [batch, (seq_len,) num_nodes, output_dim]
        return self.decoder(inputs)


# Alias for backward compatibility
WaterLevelDecoder = MultiOutputDecoder


class CLDTS(nn.Module):
    """
    Coupled Latent Digital Twin Surrogate (CL-DTS).

    Main model combining:
    - Spatial encoding via coupled 1D-2D GNN
    - Event latent c_e for event-specific calibration
    - Dynamic latent z_t for hidden physical state
    - Variational inference for uncertainty quantification
    """

    def __init__(
        self,
        # Node feature dimensions (from data)
        static_1d_dim: int,
        static_2d_dim: int,
        dynamic_1d_dim: int,
        dynamic_2d_dim: int,
        # Architecture
        hidden_dim: int = 64,
        latent_dim: int = 32,
        event_latent_dim: int = 16,
        num_gnn_layers: int = 3,
        num_temporal_layers: int = 2,
        # Options
        use_attention: bool = True,
        use_event_latent: bool = True,
        use_dynamic_latent: bool = False,  # Start deterministic, add later
        dropout: float = 0.1,
    ):
        """
        Args:
            static_1d_dim: Dimension of static 1D node features
            static_2d_dim: Dimension of static 2D node features
            dynamic_1d_dim: Dimension of dynamic 1D node features
            dynamic_2d_dim: Dimension of dynamic 2D node features
            hidden_dim: Hidden dimension throughout
            latent_dim: Dimension of dynamic latent z_t
            event_latent_dim: Dimension of event latent c_e
            num_gnn_layers: Number of GNN layers
            num_temporal_layers: Number of temporal layers
            use_attention: Whether to use attention in GNN
            use_event_latent: Whether to use event latent c_e
            use_dynamic_latent: Whether to use dynamic latent z_t
            dropout: Dropout rate
        """
        super().__init__()
        self.use_event_latent = use_event_latent
        self.use_dynamic_latent = use_dynamic_latent
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.event_latent_dim = event_latent_dim if use_event_latent else 0

        # Spatial encoder (processes static graph)
        self.spatial_encoder_1d = SpatialEncoder(
            static_1d_dim, static_2d_dim,
            hidden_channels=hidden_dim,
            num_layers=num_gnn_layers,
            use_attention=use_attention,
            dropout=dropout,
        )
        self.spatial_encoder_2d = self.spatial_encoder_1d  # Share weights

        # Spatiotemporal encoder (combines spatial + dynamic features)
        self.st_encoder_1d = SpatioTemporalEncoder(
            spatial_dim=hidden_dim,
            dynamic_dim=dynamic_1d_dim,
            hidden_dim=hidden_dim,
            num_temporal_layers=num_temporal_layers,
            dropout=dropout,
        )
        self.st_encoder_2d = SpatioTemporalEncoder(
            spatial_dim=hidden_dim,
            dynamic_dim=dynamic_2d_dim,
            hidden_dim=hidden_dim,
            num_temporal_layers=num_temporal_layers,
            dropout=dropout,
        )

        # Event latent encoder (optional)
        if use_event_latent:
            # Encode from both 1D and 2D observations
            self.event_encoder = EventLatentEncoder(
                input_dim=dynamic_1d_dim + dynamic_2d_dim,  # Simplified: use pooled features
                hidden_dim=hidden_dim,
                latent_dim=event_latent_dim,
            )
            # Prior for event latent
            self.event_prior_mean = nn.Parameter(torch.zeros(event_latent_dim))
            self.event_prior_logvar = nn.Parameter(torch.zeros(event_latent_dim))

        # Dynamic latent components (optional)
        if use_dynamic_latent:
            self.dynamic_transition = DynamicLatentTransition(
                latent_dim=latent_dim,
                obs_dim=hidden_dim,
                event_latent_dim=self.event_latent_dim,
                hidden_dim=hidden_dim,
            )
            self.inference_net = InferenceNetwork(
                obs_dim=hidden_dim,
                event_latent_dim=self.event_latent_dim,
                hidden_dim=hidden_dim,
                latent_dim=latent_dim,
            )

        # Multi-output decoders for true autoregressive prediction
        # 1D outputs: [water_level, inlet_flow] = 2 features
        # 2D outputs: [water_level, water_volume] = 2 features (rainfall is external)
        decoder_latent_dim = latent_dim if use_dynamic_latent else hidden_dim

        # Store output dimensions for use in training/inference
        self.output_dim_1d = dynamic_1d_dim  # Predict all dynamic 1D features
        self.output_dim_2d = dynamic_2d_dim - 1  # Predict water_level + water_volume (not rainfall)

        self.decoder_1d = MultiOutputDecoder(
            latent_dim=decoder_latent_dim,
            spatial_dim=hidden_dim,
            event_latent_dim=self.event_latent_dim,
            hidden_dim=hidden_dim,
            output_dim=self.output_dim_1d,  # water_level + inlet_flow
        )
        self.decoder_2d = MultiOutputDecoder(
            latent_dim=decoder_latent_dim,
            spatial_dim=hidden_dim,
            event_latent_dim=self.event_latent_dim,
            hidden_dim=hidden_dim,
            output_dim=self.output_dim_2d,  # water_level + water_volume
        )

    def encode_spatial(
        self,
        graph: HeteroData,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode static spatial features.

        Args:
            graph: HeteroData with static node/edge features

        Returns:
            spatial_1d: [num_1d_nodes, hidden_dim]
            spatial_2d: [num_2d_nodes, hidden_dim]
        """
        return self.spatial_encoder_1d(graph)

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

        # Pool features across nodes for simplicity
        pooled_1d = prefix_1d.mean(dim=2)  # [batch, prefix_len, dynamic_1d_dim]
        pooled_2d = prefix_2d.mean(dim=2)  # [batch, prefix_len, dynamic_2d_dim]

        # Expand to shared number of "nodes" (1)
        combined = torch.cat([pooled_1d, pooled_2d], dim=-1).unsqueeze(2)

        mean, logvar = self.event_encoder(combined)
        c_e = self.event_encoder.sample(mean, logvar)

        return c_e, mean, logvar

    def forward(
        self,
        graph: HeteroData,
        input_1d: torch.Tensor,
        input_2d: torch.Tensor,
        prefix_len: int = 8,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with teacher forcing.

        Args:
            graph: Static graph structure
            input_1d: [batch, seq_len, num_1d_nodes, dynamic_1d_dim]
            input_2d: [batch, seq_len, num_2d_nodes, dynamic_2d_dim]
            prefix_len: Length of prefix for event latent encoding

        Returns:
            Dict with predictions and latent distributions
        """
        batch, seq_len, num_1d_nodes, _ = input_1d.shape
        num_2d_nodes = input_2d.shape[2]
        device = input_1d.device

        # 1. Encode spatial features (static, shared across batch/time)
        spatial_1d, spatial_2d = self.encode_spatial(graph)

        # 2. Encode event latent from prefix
        prefix_1d = input_1d[:, :prefix_len]
        prefix_2d = input_2d[:, :prefix_len]
        c_e, c_e_mean, c_e_logvar = self.encode_event_latent(prefix_1d, prefix_2d)

        # 3. Encode spatiotemporal features
        st_out_1d, _ = self.st_encoder_1d(spatial_1d, input_1d)
        st_out_2d, _ = self.st_encoder_2d(spatial_2d, input_2d)

        # 4. Decode water levels
        if self.use_dynamic_latent:
            # TODO: Full VSSM inference - for now use ST output as latent proxy
            z_1d = st_out_1d
            z_2d = st_out_2d
        else:
            z_1d = st_out_1d
            z_2d = st_out_2d

        # Decode predictions
        pred_1d = self.decoder_1d(z_1d, spatial_1d, c_e)
        pred_2d = self.decoder_2d(z_2d, spatial_2d, c_e)

        return {
            'pred_1d': pred_1d.squeeze(-1),  # [batch, seq_len, num_1d_nodes]
            'pred_2d': pred_2d.squeeze(-1),  # [batch, seq_len, num_2d_nodes]
            'c_e': c_e,
            'c_e_mean': c_e_mean,
            'c_e_logvar': c_e_logvar,
        }

    def rollout(
        self,
        graph: HeteroData,
        input_1d: torch.Tensor,
        input_2d: torch.Tensor,
        horizon: int,
        prefix_len: int = 8,
        c_e: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Autoregressive rollout for inference.

        Args:
            graph: Static graph structure
            input_1d: Initial sequence [batch, init_len, num_1d_nodes, dynamic_1d_dim]
            input_2d: Initial sequence [batch, init_len, num_2d_nodes, dynamic_2d_dim]
            horizon: Number of steps to predict
            prefix_len: Length of prefix for event encoding
            c_e: Optional pre-computed event latent

        Returns:
            Dict with predictions over horizon
        """
        batch, init_len, num_1d_nodes, d1d = input_1d.shape
        num_2d_nodes = input_2d.shape[2]
        d2d = input_2d.shape[3]
        device = input_1d.device

        # Encode spatial
        spatial_1d, spatial_2d = self.encode_spatial(graph)

        # Encode event latent if not provided
        if c_e is None:
            prefix_1d = input_1d[:, :prefix_len]
            prefix_2d = input_2d[:, :prefix_len]
            c_e, _, _ = self.encode_event_latent(prefix_1d, prefix_2d)

        # Initialize hidden states
        h_1d = None
        h_2d = None

        # Current inputs
        curr_1d = input_1d
        curr_2d = input_2d

        all_pred_1d = []
        all_pred_2d = []

        for t in range(horizon):
            # Process current sequence
            st_out_1d, h_1d = self.st_encoder_1d(spatial_1d, curr_1d, h_1d)
            st_out_2d, h_2d = self.st_encoder_2d(spatial_2d, curr_2d, h_2d)

            # Get prediction for last timestep
            z_1d = st_out_1d[:, -1:]  # [batch, 1, num_1d_nodes, hidden]
            z_2d = st_out_2d[:, -1:]

            pred_1d = self.decoder_1d(z_1d, spatial_1d, c_e)
            pred_2d = self.decoder_2d(z_2d, spatial_2d, c_e)

            all_pred_1d.append(pred_1d.squeeze(-1))
            all_pred_2d.append(pred_2d.squeeze(-1))

            # Update inputs with prediction (autoregressive)
            # Construct next input by shifting and adding prediction
            new_1d = torch.zeros(batch, 1, num_1d_nodes, d1d, device=device)
            new_2d = torch.zeros(batch, 1, num_2d_nodes, d2d, device=device)

            # Water level is typically first feature - update it
            new_1d[:, :, :, 0] = pred_1d.squeeze(-1)
            new_2d[:, :, :, 0] = pred_2d.squeeze(-1)

            # Keep only last few timesteps + new prediction
            if curr_1d.shape[1] >= init_len:
                curr_1d = torch.cat([curr_1d[:, 1:], new_1d], dim=1)
                curr_2d = torch.cat([curr_2d[:, 1:], new_2d], dim=1)
            else:
                curr_1d = torch.cat([curr_1d, new_1d], dim=1)
                curr_2d = torch.cat([curr_2d, new_2d], dim=1)

        return {
            'pred_1d': torch.cat(all_pred_1d, dim=1),  # [batch, horizon, num_1d_nodes]
            'pred_2d': torch.cat(all_pred_2d, dim=1),  # [batch, horizon, num_2d_nodes]
            'c_e': c_e,
        }

    def optimize_event_latent(
        self,
        graph: HeteroData,
        prefix_1d: torch.Tensor,
        prefix_2d: torch.Tensor,
        target_1d: torch.Tensor,
        target_2d: torch.Tensor,
        num_steps: int = 50,
        lr: float = 0.01,
    ) -> torch.Tensor:
        """
        Test-time optimization of event latent c_e.
        This is the "digital twin tuning" step.

        Args:
            graph: Static graph
            prefix_1d/2d: Observation prefix
            target_1d/2d: Target water levels for prefix
            num_steps: Optimization steps
            lr: Learning rate

        Returns:
            Optimized c_e
        """
        # Initialize from posterior
        c_e, c_e_mean, _ = self.encode_event_latent(prefix_1d, prefix_2d)
        c_e = c_e_mean.clone().requires_grad_(True)

        optimizer = torch.optim.Adam([c_e], lr=lr)

        for step in range(num_steps):
            optimizer.zero_grad()

            # Forward with current c_e
            spatial_1d, spatial_2d = self.encode_spatial(graph)
            st_out_1d, _ = self.st_encoder_1d(spatial_1d, prefix_1d)
            st_out_2d, _ = self.st_encoder_2d(spatial_2d, prefix_2d)

            pred_1d = self.decoder_1d(st_out_1d, spatial_1d, c_e)
            pred_2d = self.decoder_2d(st_out_2d, spatial_2d, c_e)

            # Loss
            loss_1d = F.mse_loss(pred_1d.squeeze(-1), target_1d)
            loss_2d = F.mse_loss(pred_2d.squeeze(-1), target_2d)

            # Regularization toward prior
            kl_reg = 0.01 * ((c_e - self.event_prior_mean) ** 2).mean()

            loss = loss_1d + loss_2d + kl_reg
            loss.backward()
            optimizer.step()

        return c_e.detach()
