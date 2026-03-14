"""
Evaluation metrics for flood prediction.
"""

import numpy as np
import torch
from typing import Dict, Optional


def compute_rmse(
    pred: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Compute Root Mean Squared Error.

    Args:
        pred: Predictions
        target: Ground truth
        mask: Optional mask (1 = valid, 0 = ignore)

    Returns:
        RMSE value
    """
    if mask is not None:
        diff = (pred - target) ** 2
        diff = diff * mask
        mse = diff.sum() / (mask.sum() + 1e-8)
    else:
        mse = np.mean((pred - target) ** 2)

    return np.sqrt(mse)


def compute_std_rmse(
    pred: np.ndarray,
    target: np.ndarray,
    groups: np.ndarray,
) -> float:
    """
    Compute Standardized RMSE across groups (e.g., events).

    For each group, compute RMSE normalized by target std.
    Final metric is mean of per-group StdRMSE.

    Args:
        pred: Predictions
        target: Ground truth
        groups: Group indices (e.g., event IDs)

    Returns:
        StdRMSE value
    """
    unique_groups = np.unique(groups)
    std_rmses = []

    for g in unique_groups:
        mask = groups == g
        p = pred[mask]
        t = target[mask]

        rmse = np.sqrt(np.mean((p - t) ** 2))
        std = np.std(t) + 1e-8
        std_rmses.append(rmse / std)

    return np.mean(std_rmses)


def compute_peak_error(
    pred: np.ndarray,
    target: np.ndarray,
    axis: int = 0,
) -> Dict[str, float]:
    """
    Compute peak-related errors.

    Args:
        pred: Predictions [T, N] or [N]
        target: Ground truth [T, N] or [N]
        axis: Time axis

    Returns:
        Dict with peak metrics
    """
    # Peak magnitude error
    pred_peak = pred.max(axis=axis)
    target_peak = target.max(axis=axis)
    peak_error = np.mean(np.abs(pred_peak - target_peak))

    # Peak timing error (only for time series)
    if pred.ndim > 1:
        pred_peak_time = pred.argmax(axis=axis)
        target_peak_time = target.argmax(axis=axis)
        peak_timing_error = np.mean(np.abs(pred_peak_time - target_peak_time))
    else:
        peak_timing_error = 0.0

    # Peak relative error
    peak_rel_error = np.mean(np.abs(pred_peak - target_peak) / (target_peak + 1e-8))

    return {
        'peak_error': peak_error,
        'peak_timing_error': peak_timing_error,
        'peak_rel_error': peak_rel_error,
    }


def compute_recession_error(
    pred: np.ndarray,
    target: np.ndarray,
    percentile: float = 90,
) -> float:
    """
    Compute error in recession curve (post-peak).

    Args:
        pred: Predictions [T, N]
        target: Ground truth [T, N]
        percentile: Percentile threshold for peak definition

    Returns:
        Recession RMSE
    """
    # Find approximate peak time per node
    peak_threshold = np.percentile(target, percentile, axis=0)

    # Mask for recession (below threshold after peak)
    peak_time = target.argmax(axis=0)
    T = target.shape[0]

    recession_mask = np.zeros_like(target, dtype=bool)
    for n in range(target.shape[1]):
        for t in range(peak_time[n], T):
            if target[t, n] < peak_threshold[n]:
                recession_mask[t, n] = True

    if recession_mask.sum() == 0:
        return 0.0

    recession_rmse = np.sqrt(np.mean((pred[recession_mask] - target[recession_mask]) ** 2))
    return recession_rmse


class MetricTracker:
    """
    Track and aggregate metrics during evaluation.
    """

    def __init__(self):
        self.metrics = {}
        self.counts = {}

    def update(self, name: str, value: float, count: int = 1):
        """Add a metric value."""
        if name not in self.metrics:
            self.metrics[name] = 0.0
            self.counts[name] = 0
        self.metrics[name] += value * count
        self.counts[name] += count

    def compute(self) -> Dict[str, float]:
        """Compute mean of all tracked metrics."""
        return {
            name: self.metrics[name] / (self.counts[name] + 1e-8)
            for name in self.metrics
        }

    def reset(self):
        """Reset all metrics."""
        self.metrics = {}
        self.counts = {}
