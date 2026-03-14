"""
Loss functions for CL-DTS training.
Includes ELBO, rollout losses, and physics-inspired regularizers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class ELBOLoss(nn.Module):
    """
    Evidence Lower Bound (ELBO) loss for variational training.

    L = E_q[log p(y|z)] - beta * KL(q(z)|p(z))
    """

    def __init__(
        self,
        beta: float = 1.0,
        free_bits: float = 0.0,
        kl_annealing: bool = True,
    ):
        """
        Args:
            beta: KL weight (beta-VAE)
            free_bits: Minimum bits for KL (prevents posterior collapse)
            kl_annealing: Whether to anneal beta during training
        """
        super().__init__()
        self.beta = beta
        self.free_bits = free_bits
        self.kl_annealing = kl_annealing
        self._current_beta = 0.0 if kl_annealing else beta

    def update_beta(self, step: int, total_steps: int, warmup_steps: int = 1000):
        """Update beta according to annealing schedule."""
        if self.kl_annealing:
            if step < warmup_steps:
                self._current_beta = self.beta * (step / warmup_steps)
            else:
                self._current_beta = self.beta

    def kl_divergence(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        prior_mean: Optional[torch.Tensor] = None,
        prior_logvar: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute KL divergence KL(q||p) for Gaussian distributions.

        Args:
            mean: Posterior mean
            logvar: Posterior log-variance
            prior_mean: Prior mean (default: 0)
            prior_logvar: Prior log-variance (default: 0)

        Returns:
            KL divergence (scalar)
        """
        if prior_mean is None:
            prior_mean = torch.zeros_like(mean)
        if prior_logvar is None:
            prior_logvar = torch.zeros_like(logvar)

        # KL(N(mu1, sigma1) || N(mu0, sigma0))
        # = 0.5 * (log(sigma0^2/sigma1^2) + (sigma1^2 + (mu1-mu0)^2)/sigma0^2 - 1)
        var_ratio = torch.exp(logvar - prior_logvar)
        mean_diff_sq = (mean - prior_mean) ** 2 / torch.exp(prior_logvar)

        kl = 0.5 * (prior_logvar - logvar + var_ratio + mean_diff_sq - 1)

        # Free bits: clamp minimum KL per dimension
        if self.free_bits > 0:
            kl = torch.clamp(kl, min=self.free_bits)

        return kl.sum(dim=-1).mean()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mean: Optional[torch.Tensor] = None,
        logvar: Optional[torch.Tensor] = None,
        prior_mean: Optional[torch.Tensor] = None,
        prior_logvar: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute ELBO loss.

        Args:
            pred: Predictions [batch, seq_len, num_nodes]
            target: Targets [batch, seq_len, num_nodes]
            mean: Posterior mean (for KL)
            logvar: Posterior log-variance (for KL)
            prior_mean/logvar: Prior parameters

        Returns:
            loss: Total ELBO loss
            components: Dict with loss components
        """
        # Reconstruction loss (negative log-likelihood under Gaussian)
        recon_loss = F.mse_loss(pred, target)

        # KL divergence
        if mean is not None and logvar is not None:
            kl_loss = self.kl_divergence(mean, logvar, prior_mean, prior_logvar)
        else:
            kl_loss = torch.tensor(0.0, device=pred.device)

        # Total loss
        loss = recon_loss + self._current_beta * kl_loss

        return loss, {
            'total': loss.detach(),
            'recon': recon_loss.detach(),
            'kl': kl_loss.detach(),
            'beta': torch.tensor(self._current_beta),
        }


class RolloutLoss(nn.Module):
    """
    Multi-step rollout loss for training autoregressive models.
    Trains with K-step unrolling to prevent drift at inference.
    """

    def __init__(
        self,
        rollout_steps: int = 8,
        scheduled_sampling: bool = True,
        teacher_forcing_ratio: float = 1.0,
        min_teacher_forcing: float = 0.0,
    ):
        """
        Args:
            rollout_steps: Number of steps to unroll
            scheduled_sampling: Whether to decay teacher forcing
            teacher_forcing_ratio: Initial teacher forcing ratio
            min_teacher_forcing: Minimum teacher forcing ratio
        """
        super().__init__()
        self.rollout_steps = rollout_steps
        self.scheduled_sampling = scheduled_sampling
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.min_teacher_forcing = min_teacher_forcing
        self._current_tf_ratio = teacher_forcing_ratio

    def update_teacher_forcing(self, step: int, total_steps: int):
        """Decay teacher forcing ratio over training."""
        if self.scheduled_sampling:
            decay = 1 - step / total_steps
            self._current_tf_ratio = max(
                self.min_teacher_forcing,
                self.teacher_forcing_ratio * decay
            )

    def forward(
        self,
        model,
        graph,
        input_1d: torch.Tensor,
        input_2d: torch.Tensor,
        target_1d: torch.Tensor,
        target_2d: torch.Tensor,
        prefix_len: int = 8,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute rollout loss.

        Args:
            model: The CLDTS model
            graph: Static graph structure
            input_1d/2d: Input sequences
            target_1d/2d: Target sequences
            prefix_len: Prefix length for event encoding

        Returns:
            loss: Rollout loss
            components: Dict with loss components
        """
        batch, seq_len, num_1d_nodes, d1d = input_1d.shape
        num_2d_nodes = input_2d.shape[2]
        d2d = input_2d.shape[3]
        device = input_1d.device

        # Encode spatial and event latent (done once)
        spatial_1d, spatial_2d = model.encode_spatial(graph)
        prefix_1d = input_1d[:, :prefix_len]
        prefix_2d = input_2d[:, :prefix_len]
        c_e, c_e_mean, c_e_logvar = model.encode_event_latent(prefix_1d, prefix_2d)

        # Initialize hidden states
        h_1d = None
        h_2d = None

        # Current inputs for rollout
        curr_1d = input_1d[:, :prefix_len]
        curr_2d = input_2d[:, :prefix_len]

        all_losses = []

        for t in range(min(self.rollout_steps, seq_len - prefix_len)):
            # Process current sequence
            st_out_1d, h_1d = model.st_encoder_1d(spatial_1d, curr_1d, h_1d)
            st_out_2d, h_2d = model.st_encoder_2d(spatial_2d, curr_2d, h_2d)

            # Get prediction
            z_1d = st_out_1d[:, -1:]
            z_2d = st_out_2d[:, -1:]

            pred_1d = model.decoder_1d(z_1d, spatial_1d, c_e).squeeze(-1)
            pred_2d = model.decoder_2d(z_2d, spatial_2d, c_e).squeeze(-1)

            # Compute loss for this step
            target_idx = prefix_len + t
            loss_1d = F.mse_loss(pred_1d[:, 0], target_1d[:, target_idx])
            loss_2d = F.mse_loss(pred_2d[:, 0], target_2d[:, target_idx])

            all_losses.append(loss_1d + loss_2d)

            # Prepare next input (scheduled sampling)
            use_teacher = torch.rand(1).item() < self._current_tf_ratio

            if use_teacher and target_idx < seq_len:
                # Teacher forcing: use ground truth
                next_1d = input_1d[:, target_idx:target_idx+1]
                next_2d = input_2d[:, target_idx:target_idx+1]
            else:
                # Use model predictions
                next_1d = torch.zeros(batch, 1, num_1d_nodes, d1d, device=device)
                next_2d = torch.zeros(batch, 1, num_2d_nodes, d2d, device=device)
                next_1d[:, :, :, 0] = pred_1d[:, 0]
                next_2d[:, :, :, 0] = pred_2d[:, 0]

            # Slide window
            curr_1d = torch.cat([curr_1d[:, 1:], next_1d], dim=1)
            curr_2d = torch.cat([curr_2d[:, 1:], next_2d], dim=1)

        # Average loss over rollout steps
        rollout_loss = torch.stack(all_losses).mean()

        # KL loss for event latent
        kl_loss = torch.tensor(0.0, device=device)
        if model.use_event_latent:
            var_ratio = torch.exp(c_e_logvar)
            mean_diff_sq = c_e_mean ** 2
            kl_loss = 0.5 * (-c_e_logvar + var_ratio + mean_diff_sq - 1).sum(dim=-1).mean()

        loss = rollout_loss + 0.01 * kl_loss

        return loss, {
            'total': loss.detach(),
            'rollout': rollout_loss.detach(),
            'kl': kl_loss.detach(),
            'tf_ratio': torch.tensor(self._current_tf_ratio),
        }


class PhysicsRegularizer(nn.Module):
    """
    Physics-inspired regularizers for flood prediction stability.
    """

    def __init__(
        self,
        non_negative_weight: float = 0.1,
        smoothness_weight: float = 0.01,
        coupling_weight: float = 0.01,
    ):
        """
        Args:
            non_negative_weight: Weight for non-negativity penalty
            smoothness_weight: Weight for temporal smoothness
            coupling_weight: Weight for 1D-2D coupling consistency
        """
        super().__init__()
        self.non_negative_weight = non_negative_weight
        self.smoothness_weight = smoothness_weight
        self.coupling_weight = coupling_weight

    def forward(
        self,
        pred_1d: torch.Tensor,
        pred_2d: torch.Tensor,
        coupling_edges: Optional[torch.Tensor] = None,
        min_water_level: float = 0.0,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute physics regularization losses.

        Args:
            pred_1d: 1D predictions [batch, seq_len, num_1d_nodes]
            pred_2d: 2D predictions [batch, seq_len, num_2d_nodes]
            coupling_edges: 1D-2D coupling edge indices [2, num_edges]
            min_water_level: Minimum physical water level

        Returns:
            loss: Total regularization loss
            components: Dict with components
        """
        # Handle both 2D [batch, nodes] and 3D [batch, seq_len, nodes] inputs
        has_temporal = pred_1d.dim() == 3

        # 1. Non-negativity penalty (ReLU on negative values)
        neg_1d = F.relu(min_water_level - pred_1d)
        neg_2d = F.relu(min_water_level - pred_2d)
        non_neg_loss = neg_1d.mean() + neg_2d.mean()

        # 2. Temporal smoothness (second-order difference penalty)
        # Only apply if we have 3+ timesteps
        if has_temporal and pred_1d.shape[1] > 2:
            diff2_1d = pred_1d[:, 2:] - 2 * pred_1d[:, 1:-1] + pred_1d[:, :-2]
            diff2_2d = pred_2d[:, 2:] - 2 * pred_2d[:, 1:-1] + pred_2d[:, :-2]
            smooth_loss = diff2_1d.abs().mean() + diff2_2d.abs().mean()
        else:
            smooth_loss = torch.tensor(0.0, device=pred_1d.device)

        # 3. Coupling consistency (1D and coupled 2D nodes should be similar)
        coupling_loss = torch.tensor(0.0, device=pred_1d.device)
        if coupling_edges is not None and coupling_edges.shape[1] > 0:
            src_nodes = coupling_edges[0]  # 1D nodes
            dst_nodes = coupling_edges[1]  # 2D nodes

            # Get water levels at coupled nodes - handle both 2D and 3D
            if has_temporal:
                wl_1d = pred_1d[:, :, src_nodes]  # [batch, seq_len, num_edges]
                wl_2d = pred_2d[:, :, dst_nodes]  # [batch, seq_len, num_edges]
            else:
                wl_1d = pred_1d[:, src_nodes]  # [batch, num_edges]
                wl_2d = pred_2d[:, dst_nodes]  # [batch, num_edges]

            # Penalize large differences
            coupling_loss = (wl_1d - wl_2d).abs().mean()

        # Total
        loss = (
            self.non_negative_weight * non_neg_loss +
            self.smoothness_weight * smooth_loss +
            self.coupling_weight * coupling_loss
        )

        return loss, {
            'total': loss.detach(),
            'non_neg': non_neg_loss.detach(),
            'smooth': smooth_loss.detach(),
            'coupling': coupling_loss.detach(),
        }


class CombinedLoss(nn.Module):
    """
    Combined loss function for CL-DTS training.
    """

    def __init__(
        self,
        elbo_weight: float = 1.0,
        rollout_weight: float = 1.0,
        physics_weight: float = 0.1,
        beta: float = 0.1,
        rollout_steps: int = 8,
    ):
        super().__init__()
        self.elbo = ELBOLoss(beta=beta)
        self.rollout = RolloutLoss(rollout_steps=rollout_steps)
        self.physics = PhysicsRegularizer()

        self.elbo_weight = elbo_weight
        self.rollout_weight = rollout_weight
        self.physics_weight = physics_weight

    def forward(
        self,
        pred_1d: torch.Tensor,
        pred_2d: torch.Tensor,
        target_1d: torch.Tensor,
        target_2d: torch.Tensor,
        c_e_mean: Optional[torch.Tensor] = None,
        c_e_logvar: Optional[torch.Tensor] = None,
        coupling_edges: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute combined loss.
        """
        # ELBO loss for 1D
        elbo_1d, elbo_1d_comp = self.elbo(pred_1d, target_1d, c_e_mean, c_e_logvar)
        # ELBO loss for 2D
        elbo_2d, elbo_2d_comp = self.elbo(pred_2d, target_2d)

        elbo_loss = elbo_1d + elbo_2d

        # Physics regularization
        physics_loss, physics_comp = self.physics(pred_1d, pred_2d, coupling_edges)

        # Total
        loss = (
            self.elbo_weight * elbo_loss +
            self.physics_weight * physics_loss
        )

        return loss, {
            'total': loss.detach(),
            'elbo': elbo_loss.detach(),
            'elbo_1d': elbo_1d.detach(),
            'elbo_2d': elbo_2d.detach(),
            'physics': physics_loss.detach(),
            'kl': elbo_1d_comp['kl'],
        }
