"""
Evaluation metrics for Urban Flood Modelling competition.

Uses standardized RMSE with equal weighting for 1D and 2D nodes.
RMSE is standardized by the standard deviation of each (model, node_type) combination.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional


def compute_std_from_training(data_dir: str = "data") -> Dict[Tuple[int, str], float]:
    """
    Compute standard deviation of water levels for each (model_id, node_type) from training data.

    Returns:
        Dict mapping (model_id, node_type) -> std_dev
    """
    std_values = {}
    data_path = Path(data_dir)

    for model_id in [1, 2]:
        model_dir = data_path / f"Model_{model_id}"
        train_dir = model_dir / "train"

        # Collect water levels from all training events
        water_levels_1d = []
        water_levels_2d = []

        # Find all event directories
        event_dirs = sorted([d for d in train_dir.iterdir() if d.is_dir() and d.name.startswith("event_")])
        print(f"Model {model_id}: Found {len(event_dirs)} training events")

        for event_dir in event_dirs:
            # Load 1D node dynamic data
            file_1d = event_dir / "1d_nodes_dynamic_all.csv"
            if file_1d.exists():
                df_1d = pd.read_csv(file_1d)
                wl = df_1d['water_level'].dropna().values
                water_levels_1d.extend(wl)

            # Load 2D node dynamic data
            file_2d = event_dir / "2d_nodes_dynamic_all.csv"
            if file_2d.exists():
                df_2d = pd.read_csv(file_2d)
                wl = df_2d['water_level'].dropna().values
                water_levels_2d.extend(wl)

        # Compute std
        std_values[(model_id, '1d')] = float(np.std(water_levels_1d))
        std_values[(model_id, '2d')] = float(np.std(water_levels_2d))

        print(f"  1D: {len(water_levels_1d):,} samples, std = {std_values[(model_id, '1d')]:.6f}")
        print(f"  2D: {len(water_levels_2d):,} samples, std = {std_values[(model_id, '2d')]:.6f}")

    return std_values


def standardized_rmse(
    predictions: np.ndarray,
    targets: np.ndarray,
    std_dev: float
) -> float:
    """
    Compute standardized RMSE = RMSE / std_dev.

    Args:
        predictions: Predicted values
        targets: Ground truth values
        std_dev: Standard deviation for normalization

    Returns:
        Standardized RMSE
    """
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    return rmse / std_dev


def evaluate_submission(
    submission_df: pd.DataFrame,
    ground_truth_df: pd.DataFrame,
    std_values: Dict[Tuple[int, str], float]
) -> Tuple[float, Dict[str, float]]:
    """
    Evaluate submission using standardized RMSE with equal 1D/2D weighting.

    Args:
        submission_df: Submission dataframe with columns [row_id, model_id, event_id, node_type, node_id, water_level]
        ground_truth_df: Ground truth with same columns
        std_values: Dict of (model_id, node_type) -> std_dev

    Returns:
        final_score: Overall standardized RMSE (lower is better)
        component_scores: Dict with per-model, per-node-type scores
    """
    # Merge to align predictions with ground truth
    merged = submission_df.merge(
        ground_truth_df,
        on=['row_id', 'model_id', 'event_id', 'node_type', 'node_id'],
        suffixes=('_pred', '_true')
    )

    component_scores = {}
    model_scores = []

    for model_id in [1, 2]:
        model_data = merged[merged['model_id'] == model_id]

        # Compute standardized RMSE for 1D nodes
        data_1d = model_data[model_data['node_type'] == '1d']
        if len(data_1d) > 0:
            std_1d = std_values.get((model_id, '1d'), 1.0)
            score_1d = standardized_rmse(
                data_1d['water_level_pred'].values,
                data_1d['water_level_true'].values,
                std_1d
            )
            component_scores[f'model_{model_id}_1d'] = score_1d
        else:
            score_1d = 0.0

        # Compute standardized RMSE for 2D nodes
        data_2d = model_data[model_data['node_type'] == '2d']
        if len(data_2d) > 0:
            std_2d = std_values.get((model_id, '2d'), 1.0)
            score_2d = standardized_rmse(
                data_2d['water_level_pred'].values,
                data_2d['water_level_true'].values,
                std_2d
            )
            component_scores[f'model_{model_id}_2d'] = score_2d
        else:
            score_2d = 0.0

        # Equal weighting: average of 1D and 2D scores for this model
        model_score = (score_1d + score_2d) / 2.0
        component_scores[f'model_{model_id}_avg'] = model_score
        model_scores.append(model_score)

    # Final score: average across models (assuming equal model weight too)
    final_score = np.mean(model_scores)
    component_scores['final'] = final_score

    return final_score, component_scores


class StandardizedRMSELoss:
    """
    Loss function that mimics the competition metric.
    Uses standardized RMSE with equal weighting for 1D and 2D.
    """

    def __init__(self, std_1d: float, std_2d: float):
        """
        Args:
            std_1d: Standard deviation for 1D nodes
            std_2d: Standard deviation for 2D nodes
        """
        self.std_1d = std_1d
        self.std_2d = std_2d

    def __call__(
        self,
        pred_1d: np.ndarray,
        pred_2d: np.ndarray,
        target_1d: np.ndarray,
        target_2d: np.ndarray
    ) -> Tuple[float, Dict[str, float]]:
        """
        Compute standardized RMSE loss with equal weighting.

        Returns:
            loss: Combined standardized RMSE
            components: Dict with component losses
        """
        # Standardized RMSE for 1D
        mse_1d = np.mean((pred_1d - target_1d) ** 2)
        rmse_1d = np.sqrt(mse_1d)
        std_rmse_1d = rmse_1d / self.std_1d

        # Standardized RMSE for 2D
        mse_2d = np.mean((pred_2d - target_2d) ** 2)
        rmse_2d = np.sqrt(mse_2d)
        std_rmse_2d = rmse_2d / self.std_2d

        # Equal weighting
        loss = (std_rmse_1d + std_rmse_2d) / 2.0

        return loss, {
            'std_rmse_1d': std_rmse_1d,
            'std_rmse_2d': std_rmse_2d,
            'rmse_1d': rmse_1d,
            'rmse_2d': rmse_2d,
            'total': loss
        }


if __name__ == "__main__":
    # Compute std values from training data
    print("Computing standard deviations from training data...\n")
    std_values = compute_std_from_training("data")

    print("\n" + "="*50)
    print("Standard Deviation Values for Evaluation:")
    print("="*50)
    for (model_id, node_type), std in sorted(std_values.items()):
        print(f"  Model {model_id}, {node_type.upper()}: {std:.6f}")
