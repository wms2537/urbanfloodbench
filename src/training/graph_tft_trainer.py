"""
PyTorch Lightning trainer for Graph-TFT model.

Key features:
- Multi-horizon training (predict all K steps at once)
- Horizon-weighted loss (heavier weights near peaks/later horizons)
- Event latent regularization (KL divergence)
- Test-time calibration support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Optional, Any
import math

from ..models.graph_tft import GraphTFT


class GraphTFTTrainer(pl.LightningModule):
    """
    PyTorch Lightning module for training Graph-TFT.

    Uses multi-horizon prediction to avoid autoregressive error accumulation.
    """

    def __init__(
        self,
        model: GraphTFT,
        graph,  # HeteroData
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        beta: float = 0.01,  # KL weight
        horizon_weighting: str = 'linear',  # 'uniform', 'linear', 'exp'
        warmup_epochs: int = 5,
        max_epochs: int = 50,
        # Normalization stats for denormalization
        norm_stats: Optional[Dict] = None,
        # Loss options
        loss_type: str = 'mse',  # 'mse', 'huber', 'weighted_mse'
        huber_delta: float = 1.0,
        high_water_weight: float = 2.0,  # Extra weight for high water levels
    ):
        super().__init__()
        self.model = model
        self.graph = graph
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.beta = beta
        self.horizon_weighting = horizon_weighting
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.norm_stats = norm_stats
        self.loss_type = loss_type
        self.huber_delta = huber_delta
        self.high_water_weight = high_water_weight

        self.save_hyperparameters(ignore=['model', 'graph', 'norm_stats'])

    def forward(self, input_1d, input_2d, **kwargs):
        return self.model(self.graph, input_1d, input_2d, **kwargs)

    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute loss based on loss_type setting."""
        if self.loss_type == 'huber':
            return F.huber_loss(pred, target, delta=self.huber_delta)
        elif self.loss_type == 'weighted_mse':
            # Weight high water levels more (above mean + 1 std)
            weights = torch.ones_like(target)
            if self.norm_stats is not None:
                # High values get extra weight
                high_mask = target > 0.5  # Normalized, so 0.5 std above mean
                weights = torch.where(high_mask,
                                     torch.full_like(weights, self.high_water_weight),
                                     weights)
            return (weights * (pred - target) ** 2).mean()
        else:  # mse
            return F.mse_loss(pred, target)

    def _get_horizon_weights(self, horizon: int, device: torch.device) -> torch.Tensor:
        """
        Compute weights for different horizons.

        Later horizons are typically harder to predict and may deserve more weight.
        """
        if self.horizon_weighting == 'uniform':
            return torch.ones(horizon, device=device) / horizon

        elif self.horizon_weighting == 'linear':
            # Linearly increasing weight
            weights = torch.arange(1, horizon + 1, device=device, dtype=torch.float)
            return weights / weights.sum()

        elif self.horizon_weighting == 'exp':
            # Exponentially increasing weight
            weights = torch.exp(torch.arange(horizon, device=device, dtype=torch.float) * 0.05)
            return weights / weights.sum()

        else:
            return torch.ones(horizon, device=device) / horizon

    def training_step(self, batch, batch_idx):
        """
        Training step with multi-horizon prediction.

        Batch should contain:
        - input_1d: [batch, seq_len, num_1d_nodes, dynamic_1d_dim]
        - input_2d: [batch, seq_len, num_2d_nodes, dynamic_2d_dim]
        - target_1d: [batch, horizon, num_1d_nodes] (water level targets)
        - target_2d: [batch, horizon, num_2d_nodes] (water level targets)
        - future_rainfall: [batch, horizon, num_2d_nodes, 1] (optional)
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

        # Forward pass
        outputs = self.model(
            self.graph, input_1d, input_2d,
            prefix_len=prefix_len,
            future_rainfall=future_rainfall,
        )

        # Get predictions (water level is first feature)
        pred_1d = outputs['pred_1d'][..., 0]  # [batch, horizon, num_1d_nodes]
        pred_2d = outputs['pred_2d'][..., 0]  # [batch, horizon, num_2d_nodes]

        # Squeeze targets if they have an extra feature dimension
        if target_1d.dim() == 4:
            target_1d = target_1d.squeeze(-1)  # [B, H, N, 1] -> [B, H, N]
        if target_2d.dim() == 4:
            target_2d = target_2d.squeeze(-1)

        # Ensure prediction and target horizons match
        horizon = min(pred_1d.shape[1], target_1d.shape[1])
        pred_1d = pred_1d[:, :horizon]
        pred_2d = pred_2d[:, :horizon]
        target_1d = target_1d[:, :horizon]
        target_2d = target_2d[:, :horizon]

        # Compute horizon weights
        horizon_weights = self._get_horizon_weights(horizon, pred_1d.device)

        # Per-horizon losses
        losses_1d = []
        losses_2d = []
        for h in range(horizon):
            loss_1d_h = self._compute_loss(pred_1d[:, h], target_1d[:, h])
            loss_2d_h = self._compute_loss(pred_2d[:, h], target_2d[:, h])
            losses_1d.append(loss_1d_h * horizon_weights[h])
            losses_2d.append(loss_2d_h * horizon_weights[h])

        loss_1d = sum(losses_1d)
        loss_2d = sum(losses_2d)

        # KL divergence for event latent
        if self.model.use_event_latent:
            kl_loss = self._kl_divergence(
                outputs['c_e_mean'], outputs['c_e_logvar'],
                self.model.event_prior_mean, self.model.event_prior_logvar
            )
            # KL annealing: increase weight over warmup period
            kl_weight = min(1.0, self.current_epoch / max(1, self.warmup_epochs)) * self.beta
        else:
            kl_loss = torch.tensor(0.0, device=pred_1d.device)
            kl_weight = 0.0

        # Total loss
        total_loss = loss_1d + loss_2d + kl_weight * kl_loss

        # Auxiliary loss for other features (inlet_flow, water_volume)
        if self.model.output_dim_1d > 1:
            aux_pred_1d = outputs['pred_1d'][..., 1:]
            # Auxiliary targets would need to be in batch
            aux_target_1d = batch.get('aux_target_1d')
            if aux_target_1d is not None:
                aux_loss_1d = F.mse_loss(aux_pred_1d[:, :horizon], aux_target_1d[:, :horizon])
                total_loss = total_loss + 0.5 * aux_loss_1d

        if self.model.output_dim_2d > 1:
            aux_pred_2d = outputs['pred_2d'][..., 1:]
            aux_target_2d = batch.get('aux_target_2d')
            if aux_target_2d is not None:
                aux_loss_2d = F.mse_loss(aux_pred_2d[:, :horizon], aux_target_2d[:, :horizon])
                total_loss = total_loss + 0.5 * aux_loss_2d

        # Logging
        self.log('train/loss', total_loss, prog_bar=True)
        self.log('train/loss_1d', loss_1d)
        self.log('train/loss_2d', loss_2d)
        self.log('train/kl_loss', kl_loss)

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

        # Squeeze targets if they have an extra feature dimension
        if target_1d.dim() == 4:
            target_1d = target_1d.squeeze(-1)  # [B, H, N, 1] -> [B, H, N]
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

        # Standardized RMSE (divide by std)
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

        # Logging
        self.log('val/rmse_1d', rmse_1d)
        self.log('val/rmse_2d', rmse_2d)
        self.log('val/std_rmse_1d', std_rmse_1d)
        self.log('val/std_rmse_2d', std_rmse_2d)
        self.log('val/std_rmse', std_rmse, prog_bar=True)

        return {'val_loss': std_rmse}

    def _kl_divergence(
        self,
        mean_q: torch.Tensor,
        logvar_q: torch.Tensor,
        mean_p: torch.Tensor,
        logvar_p: torch.Tensor,
    ) -> torch.Tensor:
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


class MultiHorizonDataset(torch.utils.data.Dataset):
    """
    Dataset for multi-horizon prediction training.

    Each sample contains:
    - Input sequence (prefix for encoding)
    - Target sequence (all future timesteps to predict)
    - Future rainfall (known for all future timesteps)
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
        # Flatten across events and time
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

        # Targets are the future water levels
        target_seq_1d = self.data_1d[event_idx, end_prefix:end_horizon]
        target_seq_2d = self.data_2d[event_idx, end_prefix:end_horizon]

        # Water level targets
        target_1d = target_seq_1d[..., 0]  # Water level is feature 0 for 1D
        target_2d = target_seq_2d[..., 1]  # Water level is feature 1 for 2D (after rainfall)

        # Future rainfall (feature 0 for 2D)
        future_rainfall = target_seq_2d[..., 0:1]

        # Auxiliary targets
        aux_target_1d = target_seq_1d[..., 1:]  # inlet_flow etc.
        aux_target_2d = target_seq_2d[..., 2:]  # water_volume etc.

        # Normalize if needed
        if self.normalize and self.norm_stats is not None:
            input_1d = (input_1d - self.norm_stats['1d']['mean']) / self.norm_stats['1d']['std']
            input_2d = (input_2d - self.norm_stats['2d']['mean']) / self.norm_stats['2d']['std']
            target_1d = (target_1d - self.norm_stats['target_1d']['mean']) / self.norm_stats['target_1d']['std']
            target_2d = (target_2d - self.norm_stats['target_2d']['mean']) / self.norm_stats['target_2d']['std']
            # Future rainfall normalized with 2D stats (first feature)
            future_rainfall = (future_rainfall - self.norm_stats['2d']['mean'][0]) / self.norm_stats['2d']['std'][0]

        return {
            'input_1d': input_1d,
            'input_2d': input_2d,
            'target_1d': target_1d,
            'target_2d': target_2d,
            'future_rainfall': future_rainfall,
            'aux_target_1d': aux_target_1d,
            'aux_target_2d': aux_target_2d,
            'prefix_len': self.prefix_len,
        }
