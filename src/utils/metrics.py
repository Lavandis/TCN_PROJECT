from __future__ import annotations

import numpy as np


def compute_evaluation_summary(
    targets: np.ndarray,
    predictions: np.ndarray,
) -> tuple[dict[str, float | int], dict[str, np.ndarray]]:
    if targets.shape != predictions.shape:
        raise ValueError(f"Targets and predictions must share shape, got {targets.shape} vs {predictions.shape}.")

    errors = predictions - targets
    absolute_errors = np.abs(errors)
    squared_errors = errors**2

    per_sample_mae = absolute_errors.mean(axis=1)
    per_sample_mse = squared_errors.mean(axis=1)
    per_sample_rmse = np.sqrt(per_sample_mse)

    metrics = {
        "num_samples": int(targets.shape[0]),
        "prediction_steps": int(targets.shape[1]),
        "overall_mae": float(absolute_errors.mean()),
        "overall_mse": float(squared_errors.mean()),
        "overall_rmse": float(np.sqrt(squared_errors.mean())),
        "mean_sample_mae": float(per_sample_mae.mean()),
        "std_sample_mae": float(per_sample_mae.std()),
        "max_sample_mae": float(per_sample_mae.max()),
        "min_sample_mae": float(per_sample_mae.min()),
    }
    per_sample = {
        "mae": per_sample_mae,
        "mse": per_sample_mse,
        "rmse": per_sample_rmse,
    }
    return metrics, per_sample
