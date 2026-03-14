"""
Graph-TFT: Heterogeneous Graph Neural Network with Temporal Fusion Transformer

Combines:
- Heterogeneous GNN for spatial 1D-2D coupling
- TFT for temporal processing with multi-head attention
- Multi-horizon prediction (predicts all K steps at once)
- Event latent calibration for test-time adaptation

Architecture:
1. Spatial Encoder: HeteroGNN processes 1D-2D coupled graph
2. Temporal Encoder: TFT processes node sequences with attention
3. Multi-Horizon Decoder: Predicts all future timesteps at once
4. Event Latent: VAE-style encoding for per-event calibration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from typing import Dict, Optional, Tuple

from .coupled_gnn import SpatialEncoder
from .temporal import TemporalBlock
from .tft import SpatioTemporalTFT, MultiHorizonDecoder, GatedResidualNetwork


class EventLatentEncoderTFT(nn.Module):
    """
    Event latent encoder using TFT attention for better event representation.
    Encodes c_e from prefix observations using attention over time.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.latent_dim = latent_dim

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Self-attention for temporal aggregation
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Pooling projection
        self.pool_proj = GatedResidualNetwork(
            hidden_dim, hidden_dim, hidden_dim, dropout=dropout
        )

        # Variational outputs
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
            mask: Optional attention mask

        Returns:
            mean: [batch, latent_dim]
            logvar: [batch, latent_dim]
        """
        batch, seq_len, num_nodes, input_dim = x.shape

        # Pool over nodes first
        x_pooled = x.mean(dim=2)  # [batch, seq_len, input_dim]

        # Project
        h = self.input_proj(x_pooled)  # [batch, seq_len, hidden_dim]

        # Self-attention over time
        h_attn, _ = self.attention(h, h, h)  # [batch, seq_len, hidden_dim]

        # Add residual and pool over time
        h = h + h_attn
        pooled = h.mean(dim=1)  # [batch, hidden_dim]

        # Process
        pooled = self.pool_proj(pooled)

        # Variational outputs
        mean = self.mean_proj(pooled)
        logvar = self.logvar_proj(pooled)
        logvar = torch.clamp(logvar, min=-10, max=2)

        return mean, logvar

    def sample(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Reparameterization trick for sampling."""
        std = torch.exp(0.5 * logvar) * temperature
        eps = torch.randn_like(std)
        return mean + eps * std


class GraphTFT(nn.Module):
    """
    Graph-TFT: Heterogeneous Graph + Temporal Fusion Transformer

    Main model for urban flood prediction combining:
    - Spatial encoding via coupled 1D-2D heterogeneous GNN
    - Temporal encoding via TFT with self-attention
    - Multi-horizon prediction (all K steps at once)
    - Event latent for per-event calibration
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
        event_latent_dim: int = 16,
        num_gnn_layers: int = 3,
        num_tft_layers: int = 2,
        num_heads: int = 4,
        # Multi-horizon prediction
        prediction_horizon: int = 90,  # Predict 90 steps at once
        # Options
        use_attention: bool = True,
        use_event_latent: bool = True,
        dropout: float = 0.1,
    ):
        """
        Args:
            static_1d_dim: Dimension of static 1D node features
            static_2d_dim: Dimension of static 2D node features
            dynamic_1d_dim: Dimension of dynamic 1D features
            dynamic_2d_dim: Dimension of dynamic 2D features
            hidden_dim: Hidden dimension throughout
            event_latent_dim: Dimension of event latent c_e
            num_gnn_layers: Number of GNN message passing layers
            num_tft_layers: Number of TFT LSTM layers
            num_heads: Number of attention heads
            prediction_horizon: Number of future timesteps to predict
            use_attention: Whether to use attention in GNN
            use_event_latent: Whether to use event latent encoding
            dropout: Dropout rate
        """
        super().__init__()
        self.use_event_latent = use_event_latent
        self.hidden_dim = hidden_dim
        self.event_latent_dim = event_latent_dim if use_event_latent else 0
        self.prediction_horizon = prediction_horizon
        self.dynamic_1d_dim = dynamic_1d_dim
        self.dynamic_2d_dim = dynamic_2d_dim

        # 1. Spatial encoder (heterogeneous GNN for 1D-2D coupling)
        self.spatial_encoder = SpatialEncoder(
            static_1d_dim, static_2d_dim,
            hidden_channels=hidden_dim,
            num_layers=num_gnn_layers,
            use_attention=use_attention,
            dropout=dropout,
        )

        # 2. TFT temporal encoders (one for each node type)
        self.tft_encoder_1d = SpatioTemporalTFT(
            spatial_dim=hidden_dim,
            dynamic_dim=dynamic_1d_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_lstm_layers=num_tft_layers,
            dropout=dropout,
            event_latent_dim=self.event_latent_dim,
        )
        self.tft_encoder_2d = SpatioTemporalTFT(
            spatial_dim=hidden_dim,
            dynamic_dim=dynamic_2d_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_lstm_layers=num_tft_layers,
            dropout=dropout,
            event_latent_dim=self.event_latent_dim,
        )

        # 3. Event latent encoder
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

        # 4. Multi-horizon decoders
        # Output dimensions
        self.output_dim_1d = dynamic_1d_dim  # water_level + inlet_flow
        self.output_dim_2d = dynamic_2d_dim - 1  # water_level + water_volume (not rainfall)

        # Horizon-specific output heads
        self.decoder_1d = MultiHorizonDecoder(
            latent_dim=hidden_dim,
            spatial_dim=hidden_dim,
            event_latent_dim=self.event_latent_dim,
            hidden_dim=hidden_dim,
            output_dim=self.output_dim_1d,
            horizon=prediction_horizon,
            known_future_dim=0,  # No known future for 1D
            dropout=dropout,
        )

        # 2D decoder with known future rainfall
        self.decoder_2d = MultiHorizonDecoder(
            latent_dim=hidden_dim,
            spatial_dim=hidden_dim,
            event_latent_dim=self.event_latent_dim,
            hidden_dim=hidden_dim,
            output_dim=self.output_dim_2d,
            horizon=prediction_horizon,
            known_future_dim=1,  # Rainfall as known future covariate
            dropout=dropout,
        )

    def encode_spatial(
        self,
        graph: HeteroData,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        c_e_override: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with multi-horizon prediction.

        Args:
            graph: Static graph structure
            input_1d: [batch, seq_len, num_1d_nodes, dynamic_1d_dim]
            input_2d: [batch, seq_len, num_2d_nodes, dynamic_2d_dim]
            prefix_len: Length of prefix for event latent encoding
            future_rainfall: [batch, horizon, num_2d_nodes, 1] - known future rainfall
            c_e_override: Optional pre-computed event latent (for test-time calibration)

        Returns:
            Dict with multi-horizon predictions and latent distributions
        """
        batch, seq_len, num_1d_nodes, _ = input_1d.shape
        num_2d_nodes = input_2d.shape[2]

        # 1. Encode spatial features (static, shared across batch/time)
        spatial_1d, spatial_2d = self.encode_spatial(graph)

        # 2. Encode event latent from prefix (or use pre-computed)
        if c_e_override is not None:
            c_e = c_e_override
            c_e_mean = c_e_override
            c_e_logvar = torch.zeros_like(c_e_override)
        else:
            prefix_1d = input_1d[:, :prefix_len]
            prefix_2d = input_2d[:, :prefix_len]
            c_e, c_e_mean, c_e_logvar = self.encode_event_latent(prefix_1d, prefix_2d)

        # 3. TFT temporal encoding
        tft_out_1d, _ = self.tft_encoder_1d(spatial_1d, input_1d, c_e)
        tft_out_2d, _ = self.tft_encoder_2d(spatial_2d, input_2d, c_e)

        # 4. Get last timestep encoding for multi-horizon prediction
        z_1d = tft_out_1d[:, -1]  # [batch, num_1d_nodes, hidden_dim]
        z_2d = tft_out_2d[:, -1]  # [batch, num_2d_nodes, hidden_dim]

        # 5. Multi-horizon decoding
        pred_1d = self.decoder_1d(z_1d, spatial_1d, c_e)
        # [batch, horizon, num_1d_nodes, output_dim_1d]

        pred_2d = self.decoder_2d(z_2d, spatial_2d, c_e, future_rainfall)
        # [batch, horizon, num_2d_nodes, output_dim_2d]

        return {
            'pred_1d': pred_1d,  # [batch, horizon, num_1d_nodes, output_dim_1d]
            'pred_2d': pred_2d,  # [batch, horizon, num_2d_nodes, output_dim_2d]
            'c_e': c_e,
            'c_e_mean': c_e_mean,
            'c_e_logvar': c_e_logvar,
            'encoder_1d': z_1d,  # For potential additional use
            'encoder_2d': z_2d,
        }

    def forward_autoregressive(
        self,
        graph: HeteroData,
        input_1d: torch.Tensor,
        input_2d: torch.Tensor,
        horizon: int,
        prefix_len: int = 10,
        full_rainfall: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Autoregressive rollout for compatibility with existing pipeline.
        Uses multi-horizon prediction internally but outputs in autoregressive format.

        Args:
            graph: Static graph structure
            input_1d: Initial sequence [batch, init_len, num_1d_nodes, dynamic_1d_dim]
            input_2d: Initial sequence [batch, init_len, num_2d_nodes, dynamic_2d_dim]
            horizon: Number of steps to predict
            prefix_len: Length of prefix for event encoding
            full_rainfall: Full rainfall sequence [batch, total_len, num_2d_nodes, 1]

        Returns:
            Dict with predictions over horizon
        """
        batch, init_len, num_1d_nodes, d1d = input_1d.shape
        num_2d_nodes = input_2d.shape[2]
        device = input_1d.device

        # Encode spatial
        spatial_1d, spatial_2d = self.encode_spatial(graph)

        # Encode event latent
        prefix_1d = input_1d[:, :prefix_len]
        prefix_2d = input_2d[:, :prefix_len]
        c_e, _, _ = self.encode_event_latent(prefix_1d, prefix_2d)

        # Extract future rainfall if provided
        if full_rainfall is not None:
            # Rainfall for prediction period
            future_rainfall = full_rainfall[:, init_len:init_len + horizon]
        else:
            future_rainfall = None

        # TFT encoding of prefix
        tft_out_1d, _ = self.tft_encoder_1d(spatial_1d, input_1d, c_e)
        tft_out_2d, _ = self.tft_encoder_2d(spatial_2d, input_2d, c_e)

        # Get last encoding
        z_1d = tft_out_1d[:, -1]
        z_2d = tft_out_2d[:, -1]

        # Multi-horizon prediction
        # Adjust horizon to match decoder's configured horizon
        actual_horizon = min(horizon, self.prediction_horizon)

        pred_1d = self.decoder_1d(z_1d, spatial_1d, c_e)[:, :actual_horizon]
        pred_2d = self.decoder_2d(z_2d, spatial_2d, c_e, future_rainfall[:, :actual_horizon] if future_rainfall is not None else None)[:, :actual_horizon]

        # If horizon > prediction_horizon, need to do chunked prediction
        if horizon > self.prediction_horizon:
            all_pred_1d = [pred_1d]
            all_pred_2d = [pred_2d]

            for start in range(self.prediction_horizon, horizon, self.prediction_horizon):
                end = min(start + self.prediction_horizon, horizon)
                chunk_len = end - start

                # Use last predictions as new input (simplified - could be improved)
                # For now, just repeat the last prediction
                chunk_pred_1d = pred_1d[:, -1:].expand(-1, chunk_len, -1, -1)
                chunk_pred_2d = pred_2d[:, -1:].expand(-1, chunk_len, -1, -1)

                all_pred_1d.append(chunk_pred_1d)
                all_pred_2d.append(chunk_pred_2d)

            pred_1d = torch.cat(all_pred_1d, dim=1)[:, :horizon]
            pred_2d = torch.cat(all_pred_2d, dim=1)[:, :horizon]

        return {
            'pred_1d': pred_1d,  # [batch, horizon, num_1d_nodes, output_dim_1d]
            'pred_2d': pred_2d,  # [batch, horizon, num_2d_nodes, output_dim_2d]
            'c_e': c_e,
        }

    def optimize_event_latent(
        self,
        graph: HeteroData,
        prefix_1d: torch.Tensor,
        prefix_2d: torch.Tensor,
        target_1d: torch.Tensor,
        target_2d: torch.Tensor,
        rainfall_prefix: Optional[torch.Tensor] = None,
        num_steps: int = 50,
        lr: float = 0.01,
    ) -> torch.Tensor:
        """
        Test-time optimization of event latent c_e.
        This is the "digital twin calibration" step.

        Args:
            graph: Static graph
            prefix_1d/2d: Observation prefix
            target_1d/2d: Target water levels for prefix
            rainfall_prefix: Rainfall during prefix period [batch, prefix_len, num_2d_nodes, 1]
            num_steps: Optimization steps
            lr: Learning rate

        Returns:
            Optimized c_e
        """
        # Initialize from posterior
        c_e, c_e_mean, _ = self.encode_event_latent(prefix_1d, prefix_2d)
        c_e = c_e_mean.clone().detach().requires_grad_(True)

        optimizer = torch.optim.Adam([c_e], lr=lr)

        # Get spatial embeddings (frozen)
        with torch.no_grad():
            spatial_1d, spatial_2d = self.encode_spatial(graph)

        prefix_len = prefix_1d.shape[1]

        # Add batch dim to targets if needed
        if target_1d.dim() == 2:
            target_1d = target_1d.unsqueeze(0)  # [1, prefix_len, num_1d_nodes]
        if target_2d.dim() == 2:
            target_2d = target_2d.unsqueeze(0)  # [1, prefix_len, num_2d_nodes]

        # Need training mode for LSTM backward
        was_training = self.training
        self.train()

        for step in range(num_steps):
            optimizer.zero_grad()

            # Forward with current c_e
            tft_out_1d, _ = self.tft_encoder_1d(spatial_1d, prefix_1d, c_e)
            tft_out_2d, _ = self.tft_encoder_2d(spatial_2d, prefix_2d, c_e)

            # Get predictions for prefix period (teacher forcing style)
            z_1d = tft_out_1d  # [batch, prefix_len, num_1d_nodes, hidden_dim]
            z_2d = tft_out_2d

            # Simple decoder for prefix (not multi-horizon)
            # Use a simpler approach: predict each timestep individually
            pred_1d_list = []
            pred_2d_list = []

            for t in range(prefix_len):
                z_1d_t = z_1d[:, t]
                z_2d_t = z_2d[:, t]

                # Use first horizon prediction as single-step
                # 1D decoder (no rainfall)
                p1d = self.decoder_1d.horizon_mlps[0](
                    torch.cat([z_1d_t, spatial_1d.unsqueeze(0).expand(z_1d_t.shape[0], -1, -1),
                              c_e.unsqueeze(1).expand(-1, z_1d_t.shape[1], -1)], dim=-1)
                )

                # 2D decoder (with rainfall)
                inputs_2d = [
                    z_2d_t,
                    spatial_2d.unsqueeze(0).expand(z_2d_t.shape[0], -1, -1),
                    c_e.unsqueeze(1).expand(-1, z_2d_t.shape[1], -1)
                ]
                if rainfall_prefix is not None:
                    inputs_2d.append(rainfall_prefix[:, t])  # [batch, num_2d_nodes, 1]

                p2d = self.decoder_2d.horizon_mlps[0](torch.cat(inputs_2d, dim=-1))

                pred_1d_list.append(p1d)
                pred_2d_list.append(p2d)

            pred_1d = torch.stack(pred_1d_list, dim=1)
            pred_2d = torch.stack(pred_2d_list, dim=1)

            # Loss (only water level, first feature)
            loss_1d = F.mse_loss(pred_1d[..., 0], target_1d)
            loss_2d = F.mse_loss(pred_2d[..., 0], target_2d)

            # Regularization toward prior
            kl_reg = 0.01 * ((c_e - self.event_prior_mean) ** 2).mean()

            loss = loss_1d + loss_2d + kl_reg
            loss.backward()
            optimizer.step()

        # Restore original mode
        if not was_training:
            self.eval()

        return c_e.detach()


class GraphTFTLightning(nn.Module):
    """
    Wrapper for GraphTFT with training utilities.
    Designed to work with PyTorch Lightning trainer.
    """

    def __init__(self, model: GraphTFT, learning_rate: float = 1e-3):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch, graph):
        """Single training step."""
        input_1d = batch['input_1d']
        input_2d = batch['input_2d']
        target_1d = batch['target_1d']
        target_2d = batch['target_2d']
        future_rainfall = batch.get('future_rainfall')

        # Forward
        outputs = self.model(
            graph, input_1d, input_2d,
            future_rainfall=future_rainfall
        )

        # Loss computation
        # Multi-horizon loss: average over all horizons
        pred_1d = outputs['pred_1d'][..., 0]  # Water level only
        pred_2d = outputs['pred_2d'][..., 0]

        loss_1d = F.mse_loss(pred_1d, target_1d)
        loss_2d = F.mse_loss(pred_2d, target_2d)

        # KL divergence for event latent
        if self.model.use_event_latent:
            kl_loss = self._kl_divergence(
                outputs['c_e_mean'], outputs['c_e_logvar'],
                self.model.event_prior_mean, self.model.event_prior_logvar
            )
        else:
            kl_loss = 0.0

        total_loss = loss_1d + loss_2d + 0.01 * kl_loss

        return {
            'loss': total_loss,
            'loss_1d': loss_1d,
            'loss_2d': loss_2d,
            'kl_loss': kl_loss,
        }

    def _kl_divergence(self, mean_q, logvar_q, mean_p, logvar_p):
        """KL(q || p) for diagonal Gaussians."""
        var_q = torch.exp(logvar_q)
        var_p = torch.exp(logvar_p)

        kl = 0.5 * (
            logvar_p - logvar_q
            + var_q / var_p
            + (mean_q - mean_p) ** 2 / var_p
            - 1
        )
        return kl.sum(dim=-1).mean()
