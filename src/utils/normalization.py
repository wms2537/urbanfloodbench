"""
Normalization utilities for flood data.
"""

import numpy as np
import torch
from typing import Dict, Optional, Union


class NodeNormalizer:
    """
    Per-node or global normalization for time series data.
    """

    def __init__(
        self,
        mode: str = 'global',  # 'global', 'per_node', 'per_event'
    ):
        """
        Args:
            mode: Normalization mode
                - 'global': Single mean/std for all data
                - 'per_node': Different mean/std per node
                - 'per_event': Different mean/std per event
        """
        self.mode = mode
        self.mean = None
        self.std = None
        self.fitted = False

    def fit(
        self,
        data: Union[np.ndarray, torch.Tensor],
        axis: Optional[tuple] = None,
    ):
        """
        Fit normalizer to data.

        Args:
            data: Data to fit on
            axis: Axes to compute stats over (None = all)
        """
        if isinstance(data, torch.Tensor):
            data = data.numpy()

        if self.mode == 'global':
            self.mean = data.mean()
            self.std = data.std() + 1e-8
        elif self.mode == 'per_node':
            # Assume data is [T, N, F] or [B, T, N, F]
            if data.ndim == 3:
                self.mean = data.mean(axis=0)  # [N, F]
                self.std = data.std(axis=0) + 1e-8
            else:
                self.mean = data.mean(axis=(0, 1))  # [N, F]
                self.std = data.std(axis=(0, 1)) + 1e-8
        else:  # per_event
            self.mean = data.mean(axis=tuple(range(1, data.ndim)))  # [B]
            self.std = data.std(axis=tuple(range(1, data.ndim))) + 1e-8

        self.fitted = True

    def transform(
        self,
        data: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """Normalize data."""
        if not self.fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")

        is_tensor = isinstance(data, torch.Tensor)
        if is_tensor:
            device = data.device
            data = data.cpu().numpy()

        # Apply normalization
        result = (data - self.mean) / self.std

        if is_tensor:
            result = torch.from_numpy(result).to(device)

        return result

    def inverse_transform(
        self,
        data: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        """Denormalize data."""
        if not self.fitted:
            raise ValueError("Normalizer not fitted. Call fit() first.")

        is_tensor = isinstance(data, torch.Tensor)
        if is_tensor:
            device = data.device
            data = data.cpu().numpy()

        result = data * self.std + self.mean

        if is_tensor:
            result = torch.from_numpy(result).to(device)

        return result

    def state_dict(self) -> Dict:
        """Get normalizer state for saving."""
        return {
            'mode': self.mode,
            'mean': self.mean,
            'std': self.std,
            'fitted': self.fitted,
        }

    def load_state_dict(self, state: Dict):
        """Load normalizer state."""
        self.mode = state['mode']
        self.mean = state['mean']
        self.std = state['std']
        self.fitted = state['fitted']


def create_submission_normalizer(
    train_data: Dict[str, np.ndarray],
) -> Dict[str, NodeNormalizer]:
    """
    Create normalizers for submission predictions.

    Args:
        train_data: Dict with '1d' and '2d' water level data

    Returns:
        Dict with normalizers for each node type
    """
    normalizers = {}

    for node_type in ['1d', '2d']:
        normalizer = NodeNormalizer(mode='global')
        normalizer.fit(train_data[node_type])
        normalizers[node_type] = normalizer

    return normalizers
