"""
PyTorch Lightning trainer for VGSSM model.

Key features:
- Dual KL losses: one for event latent c_e, one for initial state z_0
- KL annealing to prevent posterior collapse
- Horizon-weighted loss for multi-step prediction
- Support for test-time latent calibration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from typing import Dict, Optional, Any
import math

from ..models.vgssm import VGSSM


class VGSSMTrainer(pl.LightningModule):
    """
    PyTorch Lightning module for training VGSSM.

    Uses state-space rollout with dual KL regularization.
    """

    def __init__(
        self,
        model: VGSSM,
        graph,  # HeteroData
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        beta_ce: float = 0.01,  # KL weight for event latent
        beta_z: float = 0.001,  # KL weight for z_0
        horizon_weighting: str = 'linear',  # 'uniform', 'linear', 'exp'
        warmup_epochs: int = 5,
        max_epochs: int = 50,
        # Normalization stats for denormalization
        norm_stats: Optional[Dict] = None,
        # Loss options
        loss_type: str = 'mse',  # 'mse', 'huber'
        huber_delta: float = 1.0,
        # Free bits for KL (prevent collapse)
        free_bits_ce: float = 0.1,
        free_bits_z: float = 0.05,
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

        self.save_hyperparameters(ignore=['model', 'graph', 'norm_stats'])

    def forward(self, input_1d, input_2d, **kwargs):
        return self.model(self.graph, input_1d, input_2d, **kwargs)

    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction loss."""
        if self.loss_type == 'huber':
            return F.huber_loss(pred, target, delta=self.huber_delta)
        else:  # mse
            return F.mse_loss(pred, target)

    def _get_horizon_weights(self, horizon: int, device: torch.device) -> torch.Tensor:
        """Compute weights for different horizons."""
        if self.horizon_weighting == 'uniform':
            return torch.ones(horizon, device=device) / horizon

        elif self.horizon_weighting == 'linear':
            weights = torch.arange(1, horizon + 1, device=device, dtype=torch.float)
            return weights / weights.sum()

        elif self.horizon_weighting == 'exp':
            weights = torch.exp(torch.arange(horizon, device=device, dtype=torch.float) * 0.05)
            return weights / weights.sum()

        else:
            return torch.ones(horizon, device=device) / horizon

    def _kl_divergence(
        self,
        mean_q: torch.Tensor,
        logvar_q: torch.Tensor,
        mean_p: torch.Tensor,
        logvar_p: torch.Tensor,
        free_bits: float = 0.0,
    ) -> torch.Tensor:
        """
        KL(q || p) for diagonal Gaussians with optional free bits.

        Free bits: KL is computed per dimension, then max(kl_dim, free_bits) is applied
        before summing. This prevents KL collapse while allowing the model to use
        some latent dimensions.
        """
        var_q = torch.exp(logvar_q)
        var_p = torch.exp(logvar_p)

        # Per-dimension KL
        kl_per_dim = 0.5 * (
            logvar_p - logvar_q
            + var_q / var_p
            + (mean_q - mean_p) ** 2 / var_p
            - 1
        )

        # Apply free bits (per dimension)
        if free_bits > 0:
            kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)

        # Sum over dimensions, mean over batch
        return kl_per_dim.sum(dim=-1).mean()

    def _kl_z0(
        self,
        mu: torch.Tensor,  # [batch, num_nodes, latent_dim]
        logvar: torch.Tensor,
        free_bits: float = 0.0,
    ) -> torch.Tensor:
        """
        KL divergence for z_0 against N(0, I) prior.
        """
        # KL(N(mu, sigma^2) || N(0, 1))
        kl_per_dim = 0.5 * (-1 - logvar + mu.pow(2) + logvar.exp())

        # Apply free bits
        if free_bits > 0:
            kl_per_dim = torch.clamp(kl_per_dim, min=free_bits)

        # Sum over latent dims, mean over nodes and batch
        return kl_per_dim.sum(dim=-1).mean()

    def training_step(self, batch, batch_idx):
        """
        Training step with latent state rollout.

        Batch should contain:
        - input_1d: [batch, seq_len, num_1d_nodes, dynamic_1d_dim]
        - input_2d: [batch, seq_len, num_2d_nodes, dynamic_2d_dim]
        - target_1d: [batch, horizon, num_1d_nodes] (water level targets)
        - target_2d: [batch, horizon, num_2d_nodes] (water level targets)
        - future_rainfall: [batch, horizon, num_2d_nodes, 1]
        """
        input_1d = batch['input_1d']
        input_2d = batch['input_2d']
        target_1d = batch['target_1d']
        target_2d = batch['target_2d']
        future_rainfall = batch.get('future_rainfall')

        # Get prefix length from batch or use default
        prefix_len = batch.get('prefix_len', 10)
        if isinstance(prefix_len, torch.Tensor):
            prefix_len = prefix_len[0].item()

        # Forward pass with latent rollout
        outputs = self.model(
            self.graph, input_1d, input_2d,
            prefix_len=prefix_len,
            future_rainfall=future_rainfall,
        )

        # Get predictions (water level)
        pred_1d = outputs['pred_1d'][..., 0]  # [batch, horizon, num_1d_nodes]
        pred_2d = outputs['pred_2d'][..., 0]  # [batch, horizon, num_2d_nodes]

        # Squeeze targets if they have an extra feature dimension
        if target_1d.dim() == 4:
            target_1d = target_1d.squeeze(-1)
        if target_2d.dim() == 4:
            target_2d = target_2d.squeeze(-1)

        # Ensure horizons match
        horizon = min(pred_1d.shape[1], target_1d.shape[1])
        pred_1d = pred_1d[:, :horizon]
        pred_2d = pred_2d[:, :horizon]
        target_1d = target_1d[:, :horizon]
        target_2d = target_2d[:, :horizon]

        # Compute horizon weights
        horizon_weights = self._get_horizon_weights(horizon, pred_1d.device)

        # Per-horizon reconstruction losses
        losses_1d = []
        losses_2d = []
        for h in range(horizon):
            loss_1d_h = self._compute_loss(pred_1d[:, h], target_1d[:, h])
            loss_2d_h = self._compute_loss(pred_2d[:, h], target_2d[:, h])
            losses_1d.append(loss_1d_h * horizon_weights[h])
            losses_2d.append(loss_2d_h * horizon_weights[h])

        loss_recon_1d = sum(losses_1d)
        loss_recon_2d = sum(losses_2d)
        loss_recon = loss_recon_1d + loss_recon_2d

        # KL annealing factor
        kl_anneal = min(1.0, self.current_epoch / max(1, self.warmup_epochs))

        # KL for event latent c_e
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

        # KL for initial latent z_0
        kl_z0_1d = self._kl_z0(
            outputs['z0_mu_1d'], outputs['z0_logvar_1d'],
            free_bits=self.free_bits_z,
        )
        kl_z0_2d = self._kl_z0(
            outputs['z0_mu_2d'], outputs['z0_logvar_2d'],
            free_bits=self.free_bits_z,
        )
        kl_z0 = kl_z0_1d + kl_z0_2d
        kl_z0_weight = kl_anneal * self.beta_z

        # Total loss
        total_loss = loss_recon + kl_ce_weight * kl_ce + kl_z0_weight * kl_z0

        # Logging
        self.log('train/loss', total_loss, prog_bar=True)
        self.log('train/loss_recon', loss_recon)
        self.log('train/loss_1d', loss_recon_1d)
        self.log('train/loss_2d', loss_recon_2d)
        self.log('train/kl_ce', kl_ce)
        self.log('train/kl_z0', kl_z0)
        self.log('train/kl_anneal', kl_anneal)

        # Monitor latent statistics
        self.log('train/z_1d_norm', outputs['z0_mu_1d'].norm(dim=-1).mean())
        self.log('train/z_2d_norm', outputs['z0_mu_2d'].norm(dim=-1).mean())
        self.log('train/c_e_norm', outputs['c_e'].norm(dim=-1).mean())

        return total_loss

    def validation_step(self, batch, batch_idx):
        """Validation step with standardized RMSE metric."""
        input_1d = batch['input_1d']
        input_2d = batch['input_2d']
        target_1d = batch['target_1d']
        target_2d = batch['target_2d']
        future_rainfall = batch.get('future_rainfall')

        prefix_len = batch.get('prefix_len', 10)
        if isinstance(prefix_len, torch.Tensor):
            prefix_len = prefix_len[0].item()

        # Forward pass
        with torch.no_grad():
            outputs = self.model(
                self.graph, input_1d, input_2d,
                prefix_len=prefix_len,
                future_rainfall=future_rainfall,
            )

        # Get predictions
        pred_1d = outputs['pred_1d'][..., 0]
        pred_2d = outputs['pred_2d'][..., 0]

        # Squeeze targets
        if target_1d.dim() == 4:
            target_1d = target_1d.squeeze(-1)
        if target_2d.dim() == 4:
            target_2d = target_2d.squeeze(-1)

        horizon = min(pred_1d.shape[1], target_1d.shape[1])
        pred_1d = pred_1d[:, :horizon]
        pred_2d = pred_2d[:, :horizon]
        target_1d = target_1d[:, :horizon]
        target_2d = target_2d[:, :horizon]

        # RMSE
        rmse_1d = torch.sqrt(F.mse_loss(pred_1d, target_1d))
        rmse_2d = torch.sqrt(F.mse_loss(pred_2d, target_2d))

        # Standardized RMSE
        if self.norm_stats is not None:
            std_1d = self.norm_stats.get('target_1d', {}).get('std', 1.0)
            std_2d = self.norm_stats.get('target_2d', {}).get('std', 1.0)
            if isinstance(std_1d, torch.Tensor):
                std_1d = std_1d.item()
            if isinstance(std_2d, torch.Tensor):
                std_2d = std_2d.item()
        else:
            std_1d = std_2d = 1.0

        std_rmse_1d = rmse_1d / max(std_1d, 1e-6)
        std_rmse_2d = rmse_2d / max(std_2d, 1e-6)
        std_rmse = (std_rmse_1d + std_rmse_2d) / 2

        # Per-horizon RMSE for analysis
        per_horizon_rmse_1d = []
        per_horizon_rmse_2d = []
        for h in range(horizon):
            rmse_h_1d = torch.sqrt(F.mse_loss(pred_1d[:, h], target_1d[:, h]))
            rmse_h_2d = torch.sqrt(F.mse_loss(pred_2d[:, h], target_2d[:, h]))
            per_horizon_rmse_1d.append(rmse_h_1d)
            per_horizon_rmse_2d.append(rmse_h_2d)

        # Log early/mid/late horizon RMSE
        early_end = horizon // 3
        mid_end = 2 * horizon // 3

        early_rmse = (torch.stack(per_horizon_rmse_1d[:early_end]).mean() +
                      torch.stack(per_horizon_rmse_2d[:early_end]).mean()) / 2
        mid_rmse = (torch.stack(per_horizon_rmse_1d[early_end:mid_end]).mean() +
                    torch.stack(per_horizon_rmse_2d[early_end:mid_end]).mean()) / 2
        late_rmse = (torch.stack(per_horizon_rmse_1d[mid_end:]).mean() +
                     torch.stack(per_horizon_rmse_2d[mid_end:]).mean()) / 2

        # Logging
        self.log('val/rmse_1d', rmse_1d)
        self.log('val/rmse_2d', rmse_2d)
        self.log('val/std_rmse_1d', std_rmse_1d)
        self.log('val/std_rmse_2d', std_rmse_2d)
        self.log('val/std_rmse', std_rmse, prog_bar=True)
        self.log('val/rmse_early', early_rmse)
        self.log('val/rmse_mid', mid_rmse)
        self.log('val/rmse_late', late_rmse)

        return {'val_loss': std_rmse}

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.max_epochs,
            eta_min=self.learning_rate * 0.01,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            }
        }


class VGSSMDataset(torch.utils.data.Dataset):
    """
    Dataset for VGSSM training with multi-horizon targets.

    Similar to MultiHorizonDataset but optimized for state-space rollout.
    """

    def __init__(
        self,
        data_1d: torch.Tensor,  # [num_events, num_timesteps, num_1d_nodes, dynamic_1d_dim]
        data_2d: torch.Tensor,  # [num_events, num_timesteps, num_2d_nodes, dynamic_2d_dim]
        prefix_len: int = 10,
        horizon: int = 90,
        stride: int = 1,
        normalize: bool = True,
        norm_stats: Optional[Dict] = None,
    ):
        """
        Args:
            data_1d: 1D node time series for all events
            data_2d: 2D node time series for all events
            prefix_len: Number of timesteps for prefix encoding
            horizon: Number of future timesteps to predict
            stride: Stride for sliding window
            normalize: Whether to normalize data
            norm_stats: Pre-computed normalization stats
        """
        self.data_1d = data_1d
        self.data_2d = data_2d
        self.prefix_len = prefix_len
        self.horizon = horizon
        self.stride = stride
        self.normalize = normalize

        # Compute or use provided normalization stats
        if normalize:
            if norm_stats is not None:
                self.norm_stats = norm_stats
            else:
                self.norm_stats = self._compute_norm_stats()
        else:
            self.norm_stats = None

        # Create sample indices
        self.samples = self._create_samples()

    def _compute_norm_stats(self) -> Dict:
        """Compute mean and std for normalization."""
        flat_1d = self.data_1d.reshape(-1, self.data_1d.shape[-1])
        flat_2d = self.data_2d.reshape(-1, self.data_2d.shape[-1])

        return {
            '1d': {
                'mean': flat_1d.mean(dim=0),
                'std': flat_1d.std(dim=0) + 1e-8,
            },
            '2d': {
                'mean': flat_2d.mean(dim=0),
                'std': flat_2d.std(dim=0) + 1e-8,
            },
            'target_1d': {
                'mean': flat_1d[:, 0].mean(),  # Water level
                'std': flat_1d[:, 0].std() + 1e-8,
            },
            'target_2d': {
                'mean': flat_2d[:, 1].mean(),  # Water level (index 1, after rainfall)
                'std': flat_2d[:, 1].std() + 1e-8,
            },
        }

    def _create_samples(self):
        """Create list of (event_idx, start_idx) tuples."""
        samples = []
        num_events, num_timesteps = self.data_1d.shape[:2]
        required_len = self.prefix_len + self.horizon

        for event_idx in range(num_events):
            for start_idx in range(0, num_timesteps - required_len + 1, self.stride):
                samples.append((event_idx, start_idx))

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        event_idx, start_idx = self.samples[idx]
        end_prefix = start_idx + self.prefix_len
        end_horizon = end_prefix + self.horizon

        # Extract sequences
        input_1d = self.data_1d[event_idx, start_idx:end_prefix]
        input_2d = self.data_2d[event_idx, start_idx:end_prefix]

        # Targets are future water levels
        target_seq_1d = self.data_1d[event_idx, end_prefix:end_horizon]
        target_seq_2d = self.data_2d[event_idx, end_prefix:end_horizon]

        # Water level targets
        target_1d = target_seq_1d[..., 0]  # Water level is feature 0 for 1D
        target_2d = target_seq_2d[..., 1]  # Water level is feature 1 for 2D (after rainfall)

        # Future rainfall (feature 0 for 2D)
        future_rainfall = target_seq_2d[..., 0:1]

        # Future inlet flow (feature 1 for 1D, if available)
        if self.data_1d.shape[-1] > 1:
            future_inlet_flow = target_seq_1d[..., 1:2]
        else:
            future_inlet_flow = None

        # Normalize
        if self.normalize and self.norm_stats is not None:
            input_1d = (input_1d - self.norm_stats['1d']['mean']) / self.norm_stats['1d']['std']
            input_2d = (input_2d - self.norm_stats['2d']['mean']) / self.norm_stats['2d']['std']
            target_1d = (target_1d - self.norm_stats['target_1d']['mean']) / self.norm_stats['target_1d']['std']
            target_2d = (target_2d - self.norm_stats['target_2d']['mean']) / self.norm_stats['target_2d']['std']
            future_rainfall = (future_rainfall - self.norm_stats['2d']['mean'][0]) / self.norm_stats['2d']['std'][0]

        result = {
            'input_1d': input_1d,
            'input_2d': input_2d,
            'target_1d': target_1d,
            'target_2d': target_2d,
            'future_rainfall': future_rainfall,
            'prefix_len': self.prefix_len,
        }

        if future_inlet_flow is not None:
            result['future_inlet_flow'] = future_inlet_flow

        return result
