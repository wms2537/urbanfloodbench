"""
PyTorch Lightning Trainer for CL-DTS model.

Key training improvements for autoregressive flood prediction:
1. Multi-step rollout training (not just 1-step) to prevent overfitting
2. Teacher forcing with actual rainfall during rollout
3. Scheduled sampling to gradually use model predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_geometric.data import HeteroData
from typing import Dict, Optional, List, Any
import numpy as np

from ..models.cldts import CLDTS
from .losses import CombinedLoss


class FloodTrainer(pl.LightningModule):
    """
    PyTorch Lightning module for training CL-DTS.

    Uses standardized RMSE with equal weighting for 1D and 2D nodes
    to match the competition evaluation metric.

    Training strategy:
    - Multi-step rollout (8 steps) instead of 1-step prediction
    - Teacher forcing: use actual rainfall from data during rollout
    - Scheduled sampling: gradually decay teacher forcing for water_level
    """

    def __init__(
        self,
        # Model architecture
        static_1d_dim: int,
        static_2d_dim: int,
        dynamic_1d_dim: int,
        dynamic_2d_dim: int,
        hidden_dim: int = 64,
        latent_dim: int = 32,
        event_latent_dim: int = 16,
        num_gnn_layers: int = 3,
        num_temporal_layers: int = 2,
        use_attention: bool = True,
        use_event_latent: bool = True,
        use_dynamic_latent: bool = False,
        dropout: float = 0.1,
        # Training
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        beta: float = 0.1,
        rollout_steps: int = 8,
        # Scheduled sampling
        initial_teacher_forcing: float = 1.0,
        min_teacher_forcing: float = 0.0,
        teacher_forcing_decay_epochs: int = 30,
        # Data
        seq_len: int = 16,
        pred_len: int = 1,
        prefix_len: int = 8,
        # Standardization for evaluation metric (std per node type)
        std_1d: float = 1.0,
        std_2d: float = 1.0,
        # Graph (will be set by setup)
        graph: Optional[HeteroData] = None,
        normalization_stats: Optional[Dict] = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['graph', 'normalization_stats'])

        # Scheduled sampling parameters
        self.initial_teacher_forcing = initial_teacher_forcing
        self.min_teacher_forcing = min_teacher_forcing
        self.teacher_forcing_decay_epochs = teacher_forcing_decay_epochs
        self.current_teacher_forcing = initial_teacher_forcing

        # Build model
        self.model = CLDTS(
            static_1d_dim=static_1d_dim,
            static_2d_dim=static_2d_dim,
            dynamic_1d_dim=dynamic_1d_dim,
            dynamic_2d_dim=dynamic_2d_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            event_latent_dim=event_latent_dim,
            num_gnn_layers=num_gnn_layers,
            num_temporal_layers=num_temporal_layers,
            use_attention=use_attention,
            use_event_latent=use_event_latent,
            use_dynamic_latent=use_dynamic_latent,
            dropout=dropout,
        )

        # Loss
        self.loss_fn = CombinedLoss(
            beta=beta,
            rollout_steps=rollout_steps,
        )

        # Store graph and normalization
        self.graph = graph
        self.normalization_stats = normalization_stats

        # Training params
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.prefix_len = prefix_len

        # Standardization values for competition metric
        self.std_1d = std_1d
        self.std_2d = std_2d

    def set_graph(self, graph: HeteroData):
        """Set the graph structure (call before training)."""
        self.graph = graph

    def set_normalization_stats(self, stats: Dict):
        """Set normalization statistics."""
        self.normalization_stats = stats

    def forward(
        self,
        input_1d: torch.Tensor,
        input_2d: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        if self.graph is None:
            raise ValueError("Graph not set. Call set_graph() first.")

        # Move graph to device
        graph = self.graph.to(self.device)

        return self.model(graph, input_1d, input_2d, prefix_len=self.prefix_len)

    def on_train_epoch_start(self):
        """Update teacher forcing ratio at the start of each epoch."""
        if self.current_epoch < self.teacher_forcing_decay_epochs:
            # Linear decay from initial to min
            decay_progress = self.current_epoch / self.teacher_forcing_decay_epochs
            self.current_teacher_forcing = self.initial_teacher_forcing - \
                (self.initial_teacher_forcing - self.min_teacher_forcing) * decay_progress
        else:
            self.current_teacher_forcing = self.min_teacher_forcing
        self.log('train/teacher_forcing', self.current_teacher_forcing)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        """
        Training step with multi-step rollout.

        Key improvements over simple 1-step prediction:
        1. Rolls out for multiple steps (rollout_steps=8 by default)
        2. Uses ACTUAL rainfall from data (teacher forcing for rainfall)
        3. Scheduled sampling: uses model predictions with probability (1 - teacher_forcing)
        """
        input_1d = batch['input_1d']  # [batch, seq_len, N1, F1]
        input_2d = batch['input_2d']  # [batch, seq_len, N2, F2]
        target_1d = batch['target_1d']  # [batch, pred_len, N1, 1]
        target_2d = batch['target_2d']  # [batch, pred_len, N2, 1]

        batch_size, seq_len, num_1d_nodes, d1d = input_1d.shape
        num_2d_nodes = input_2d.shape[2]
        d2d = input_2d.shape[3]
        device = input_1d.device

        # Move graph to device
        graph = self.graph.to(device)

        # 1. Encode spatial features (static, shared across batch/time)
        spatial_1d, spatial_2d = self.model.encode_spatial(graph)

        # 2. Encode event latent from prefix
        prefix_1d = input_1d[:, :self.prefix_len]
        prefix_2d = input_2d[:, :self.prefix_len]
        c_e, c_e_mean, c_e_logvar = self.model.encode_event_latent(prefix_1d, prefix_2d)

        # 3. Initialize hidden states from prefix
        h_1d = None
        h_2d = None

        # Process prefix to initialize hidden states
        st_out_1d, h_1d = self.model.st_encoder_1d(spatial_1d, prefix_1d, h_1d)
        st_out_2d, h_2d = self.model.st_encoder_2d(spatial_2d, prefix_2d, h_2d)

        # Current input starts from last prefix timestep
        curr_input_1d = input_1d[:, self.prefix_len - 1:self.prefix_len]  # [batch, 1, N1, F1]
        curr_input_2d = input_2d[:, self.prefix_len - 1:self.prefix_len]  # [batch, 1, N2, F2]

        # Get current latent state
        curr_z_1d = st_out_1d[:, -1:]  # [batch, 1, N1, hidden]
        curr_z_2d = st_out_2d[:, -1:]  # [batch, 1, N2, hidden]

        # 4. Multi-step rollout with TRUE AUTOREGRESSIVE prediction
        # Decoder outputs ALL features needed for next step
        # 1D: [water_level, inlet_flow], 2D: [water_level, water_volume]
        rollout_steps = min(self.hparams.rollout_steps, seq_len - self.prefix_len)
        all_losses = []

        for t in range(rollout_steps):
            # Decode ALL features from current latent state
            # pred_1d: [batch, 1, N1, output_dim_1d] where output_dim_1d = d1d (water_level + inlet_flow)
            # pred_2d: [batch, 1, N2, output_dim_2d] where output_dim_2d = d2d-1 (water_level + water_volume)
            pred_1d = self.model.decoder_1d(curr_z_1d, spatial_1d, c_e)
            pred_2d = self.model.decoder_2d(curr_z_2d, spatial_2d, c_e)

            # Target timestep index
            target_idx = self.prefix_len + t
            if target_idx < seq_len:
                # Get ALL target features from input sequence
                # 1D features: [water_level, inlet_flow]
                target_all_1d = input_1d[:, target_idx, :, :]  # [batch, N1, d1d]
                # 2D features: [water_level, water_volume] (exclude rainfall at index 0)
                target_all_2d = input_2d[:, target_idx, :, 1:]  # [batch, N2, d2d-1]
            else:
                # Use batch target for water_level only
                target_wl_1d = target_1d[:, t, :, 0] if t < target_1d.shape[1] else target_all_1d[:, :, 0]
                target_wl_2d = target_2d[:, t, :, 0] if t < target_2d.shape[1] else target_all_2d[:, :, 0]
                target_all_1d = target_all_1d.clone()
                target_all_2d = target_all_2d.clone()
                target_all_1d[:, :, 0] = target_wl_1d
                target_all_2d[:, :, 0] = target_wl_2d

            # Compute loss on ALL predicted features
            # Primary loss: water_level (scored in competition)
            # Auxiliary loss: inlet_flow, water_volume (helps autoregressive prediction)
            pred_1d_squeezed = pred_1d.squeeze(1)  # [batch, N1, output_dim_1d]
            pred_2d_squeezed = pred_2d.squeeze(1)  # [batch, N2, output_dim_2d]

            # Water level loss (primary - competition metric)
            loss_wl_1d = F.mse_loss(pred_1d_squeezed[:, :, 0], target_all_1d[:, :, 0])
            loss_wl_2d = F.mse_loss(pred_2d_squeezed[:, :, 0], target_all_2d[:, :, 0])

            # Auxiliary losses (help autoregressive accuracy)
            loss_aux_1d = F.mse_loss(pred_1d_squeezed[:, :, 1:], target_all_1d[:, :, 1:]) if d1d > 1 else 0
            loss_aux_2d = F.mse_loss(pred_2d_squeezed[:, :, 1:], target_all_2d[:, :, 1:]) if d2d > 2 else 0

            # Combined loss: primary (1.0) + auxiliary (0.5)
            step_loss = (loss_wl_1d + loss_wl_2d) + 0.5 * (loss_aux_1d + loss_aux_2d)
            all_losses.append(step_loss)

            # 5. Prepare next input - TRUE AUTOREGRESSIVE (use predictions)
            use_teacher = torch.rand(1).item() < self.current_teacher_forcing

            # Create next input
            next_input_1d = torch.zeros(batch_size, 1, num_1d_nodes, d1d, device=device)
            next_input_2d = torch.zeros(batch_size, 1, num_2d_nodes, d2d, device=device)

            if use_teacher and target_idx < seq_len:
                # Teacher forcing: use ground truth for ALL features
                next_input_1d[:, 0, :, :] = input_1d[:, target_idx, :, :]
                next_input_2d[:, 0, :, :] = input_2d[:, target_idx, :, :]
            else:
                # TRUE AUTOREGRESSIVE: use ALL predicted features
                # 1D: directly use all predictions [water_level, inlet_flow]
                next_input_1d[:, 0, :, :] = pred_1d_squeezed

                # 2D: use actual rainfall + predicted [water_level, water_volume]
                if target_idx < seq_len:
                    next_input_2d[:, 0, :, 0] = input_2d[:, target_idx, :, 0]  # actual rainfall
                else:
                    next_input_2d[:, 0, :, 0] = curr_input_2d[:, 0, :, 0]  # keep last rainfall
                next_input_2d[:, 0, :, 1:] = pred_2d_squeezed  # predicted water_level + water_volume

            # 6. Update temporal encoder hidden state (single pass - fixes double encoder bug)
            curr_z_1d, h_1d = self.model.st_encoder_1d(spatial_1d, next_input_1d, h_1d)
            curr_z_2d, h_2d = self.model.st_encoder_2d(spatial_2d, next_input_2d, h_2d)
            curr_z_1d = curr_z_1d[:, -1:]
            curr_z_2d = curr_z_2d[:, -1:]
            curr_input_1d = next_input_1d
            curr_input_2d = next_input_2d

        # Average loss over rollout steps
        rollout_loss = torch.stack(all_losses).mean()

        # KL loss for event latent
        kl_loss = torch.tensor(0.0, device=device)
        if self.model.use_event_latent and c_e_mean is not None:
            var_ratio = torch.exp(c_e_logvar)
            mean_diff_sq = c_e_mean ** 2
            kl_loss = 0.5 * (-c_e_logvar + var_ratio + mean_diff_sq - 1).sum(dim=-1).mean()

        # Total loss
        loss = rollout_loss + self.hparams.beta * kl_loss

        # Log metrics
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/rollout_loss', rollout_loss)
        self.log('train/kl_loss', kl_loss)
        self.log('train/tf_ratio', self.current_teacher_forcing)

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """
        Validation step with multi-step rollout (NO teacher forcing).

        Uses the same multi-step rollout as inference to properly evaluate
        autoregressive performance. Only uses actual rainfall from data.
        """
        input_1d = batch['input_1d']  # [batch, seq_len, N1, F1]
        input_2d = batch['input_2d']  # [batch, seq_len, N2, F2]
        target_1d = batch['target_1d']
        target_2d = batch['target_2d']

        batch_size, seq_len, num_1d_nodes, d1d = input_1d.shape
        num_2d_nodes = input_2d.shape[2]
        d2d = input_2d.shape[3]
        device = input_1d.device

        # Move graph to device
        graph = self.graph.to(device)

        # 1. Encode spatial features
        spatial_1d, spatial_2d = self.model.encode_spatial(graph)

        # 2. Encode event latent from prefix
        prefix_1d = input_1d[:, :self.prefix_len]
        prefix_2d = input_2d[:, :self.prefix_len]
        c_e, c_e_mean, c_e_logvar = self.model.encode_event_latent(prefix_1d, prefix_2d)

        # 3. Initialize hidden states from prefix
        h_1d = None
        h_2d = None
        st_out_1d, h_1d = self.model.st_encoder_1d(spatial_1d, prefix_1d, h_1d)
        st_out_2d, h_2d = self.model.st_encoder_2d(spatial_2d, prefix_2d, h_2d)

        # Current input from last prefix timestep
        curr_input_1d = input_1d[:, self.prefix_len - 1:self.prefix_len]
        curr_input_2d = input_2d[:, self.prefix_len - 1:self.prefix_len]

        curr_z_1d = st_out_1d[:, -1:]
        curr_z_2d = st_out_2d[:, -1:]

        # 4. Multi-step rollout with TRUE AUTOREGRESSIVE prediction (matches inference)
        # Use ALL predicted features as input for next step
        rollout_steps = min(self.hparams.rollout_steps, seq_len - self.prefix_len)
        all_pred_wl_1d = []  # water_level predictions only (for scoring)
        all_pred_wl_2d = []
        all_target_wl_1d = []
        all_target_wl_2d = []

        for t in range(rollout_steps):
            # Decode ALL features
            # pred_1d: [batch, 1, N1, output_dim_1d] (water_level + inlet_flow)
            # pred_2d: [batch, 1, N2, output_dim_2d] (water_level + water_volume)
            pred_1d = self.model.decoder_1d(curr_z_1d, spatial_1d, c_e)
            pred_2d = self.model.decoder_2d(curr_z_2d, spatial_2d, c_e)

            pred_1d_squeezed = pred_1d.squeeze(1)  # [batch, N1, output_dim]
            pred_2d_squeezed = pred_2d.squeeze(1)  # [batch, N2, output_dim]

            # Store water_level predictions for scoring
            all_pred_wl_1d.append(pred_1d_squeezed[:, :, 0])
            all_pred_wl_2d.append(pred_2d_squeezed[:, :, 0])

            # Get targets (water_level only for scoring)
            target_idx = self.prefix_len + t
            if target_idx < seq_len:
                all_target_wl_1d.append(input_1d[:, target_idx, :, 0])
                all_target_wl_2d.append(input_2d[:, target_idx, :, 1])
            else:
                all_target_wl_1d.append(target_1d[:, t, :, 0] if t < target_1d.shape[1] else all_target_wl_1d[-1])
                all_target_wl_2d.append(target_2d[:, t, :, 0] if t < target_2d.shape[1] else all_target_wl_2d[-1])

            # Prepare next input - TRUE AUTOREGRESSIVE (use ALL predicted features)
            next_input_1d = torch.zeros(batch_size, 1, num_1d_nodes, d1d, device=device)
            next_input_2d = torch.zeros(batch_size, 1, num_2d_nodes, d2d, device=device)

            # 1D: use all predictions [water_level, inlet_flow]
            next_input_1d[:, 0, :, :] = pred_1d_squeezed

            # 2D: use actual rainfall + predicted [water_level, water_volume]
            if target_idx < seq_len:
                next_input_2d[:, 0, :, 0] = input_2d[:, target_idx, :, 0]  # actual rainfall
            else:
                next_input_2d[:, 0, :, 0] = curr_input_2d[:, 0, :, 0]  # keep last rainfall
            next_input_2d[:, 0, :, 1:] = pred_2d_squeezed  # predicted water_level + water_volume

            # Update hidden states
            _, h_1d = self.model.st_encoder_1d(spatial_1d, next_input_1d, h_1d)
            _, h_2d = self.model.st_encoder_2d(spatial_2d, next_input_2d, h_2d)

            curr_z_1d, _ = self.model.st_encoder_1d(spatial_1d, next_input_1d, h_1d)
            curr_z_2d, _ = self.model.st_encoder_2d(spatial_2d, next_input_2d, h_2d)
            curr_z_1d = curr_z_1d[:, -1:]
            curr_z_2d = curr_z_2d[:, -1:]
            curr_input_1d = next_input_1d
            curr_input_2d = next_input_2d

        # Stack water_level predictions and targets for scoring
        all_pred_wl_1d = torch.stack(all_pred_wl_1d, dim=1)  # [batch, rollout, N1]
        all_pred_wl_2d = torch.stack(all_pred_wl_2d, dim=1)  # [batch, rollout, N2]
        all_target_wl_1d = torch.stack(all_target_wl_1d, dim=1)
        all_target_wl_2d = torch.stack(all_target_wl_2d, dim=1)

        # Compute RMSE over all rollout steps (water_level only - competition metric)
        rmse_1d = torch.sqrt(F.mse_loss(all_pred_wl_1d, all_target_wl_1d))
        rmse_2d = torch.sqrt(F.mse_loss(all_pred_wl_2d, all_target_wl_2d))

        # Compute Standardized RMSE (competition metric)
        std_rmse_1d = rmse_1d / self.std_1d
        std_rmse_2d = rmse_2d / self.std_2d

        # Equal weighting for 1D and 2D (competition metric)
        std_rmse_avg = (std_rmse_1d + std_rmse_2d) / 2.0

        # Loss for logging
        loss = rmse_1d + rmse_2d

        # Log
        self.log('val/loss', loss, prog_bar=True)
        self.log('val/rmse_1d', rmse_1d)
        self.log('val/rmse_2d', rmse_2d)
        self.log('val/std_rmse_1d', std_rmse_1d)
        self.log('val/std_rmse_2d', std_rmse_2d)
        self.log('val/std_rmse', std_rmse_avg, prog_bar=True)  # Competition metric

        return {'loss': loss, 'std_rmse': std_rmse_avg, 'rmse_1d': rmse_1d, 'rmse_2d': rmse_2d}

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Test step with rollout evaluation."""
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=10,
            T_mult=2,
            eta_min=1e-6,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }

    @torch.no_grad()
    def predict_event(
        self,
        input_1d: torch.Tensor,
        input_2d: torch.Tensor,
        horizon: int,
        optimize_event_latent: bool = True,
        target_1d: Optional[torch.Tensor] = None,
        target_2d: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict full event trajectory.

        Args:
            input_1d/2d: Initial input sequences
            horizon: Number of steps to predict
            optimize_event_latent: Whether to optimize c_e on prefix
            target_1d/2d: Optional targets for event latent optimization

        Returns:
            Dict with predictions
        """
        self.eval()
        graph = self.graph.to(self.device)

        # Optionally optimize event latent
        c_e = None
        if optimize_event_latent and target_1d is not None and self.model.use_event_latent:
            prefix_1d = input_1d[:, :self.prefix_len]
            prefix_2d = input_2d[:, :self.prefix_len]
            target_prefix_1d = target_1d[:, :self.prefix_len]
            target_prefix_2d = target_2d[:, :self.prefix_len]

            c_e = self.model.optimize_event_latent(
                graph, prefix_1d, prefix_2d,
                target_prefix_1d, target_prefix_2d,
            )

        # Rollout
        outputs = self.model.rollout(
            graph, input_1d, input_2d, horizon,
            prefix_len=self.prefix_len, c_e=c_e,
        )

        # Denormalize if stats available
        if self.normalization_stats is not None:
            outputs['pred_1d'] = self._denormalize(
                outputs['pred_1d'], self.normalization_stats['target_1d']
            )
            outputs['pred_2d'] = self._denormalize(
                outputs['pred_2d'], self.normalization_stats['target_2d']
            )

        return outputs

    def _denormalize(self, x: torch.Tensor, stats: Dict) -> torch.Tensor:
        """Denormalize predictions."""
        mean = torch.tensor(stats['mean'], device=x.device)
        std = torch.tensor(stats['std'], device=x.device)
        return x * std + mean
