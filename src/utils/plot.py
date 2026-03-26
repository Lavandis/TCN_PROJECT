from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def choose_sample_index(
    num_samples: int,
    sample_index: int | None,
    random_seed: int,
) -> int:
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    if sample_index is not None:
        if sample_index < 0 or sample_index >= num_samples:
            raise IndexError(f"sample_index {sample_index} is out of range for {num_samples} samples.")
        return int(sample_index)

    return int(np.random.default_rng(random_seed).integers(low=0, high=num_samples))


def save_trajectory_plot(
    input_sequence: np.ndarray,
    target_sequence: np.ndarray,
    prediction_sequence: np.ndarray,
    fps: int,
    output_path: str | Path,
    title: str,
    zoom_history_seconds: float,
    zoom_future_seconds: float,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    input_steps = input_sequence.shape[0]
    pred_steps = target_sequence.shape[0]

    t_input = np.arange(input_steps) / fps
    t_future = np.arange(input_steps, input_steps + pred_steps) / fps

    figure, axis = plt.subplots(figsize=(14, 6))
    axis.plot(t_input, input_sequence, color="#7f7f7f", linewidth=1.2, alpha=0.8, label="Input History")
    axis.plot(t_future, target_sequence, color="#2e8b57", linewidth=1.5, label="Ground Truth")
    axis.plot(t_future, prediction_sequence, color="#c0392b", linewidth=1.5, linestyle="--", label="Prediction")
    axis.set_title(title)
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Angle (rad)")
    axis.grid(alpha=0.25)
    axis.legend()

    inset = axis.inset_axes([0.58, 0.5, 0.36, 0.38])
    full_truth = np.concatenate([input_sequence, target_sequence], axis=0)
    full_prediction = np.concatenate([np.full_like(input_sequence, np.nan), prediction_sequence], axis=0)
    t_full = np.arange(input_steps + pred_steps) / fps

    zoom_left = max(0, input_steps - int(round(zoom_history_seconds * fps)))
    zoom_right = min(input_steps + pred_steps, input_steps + int(round(zoom_future_seconds * fps)))
    inset.plot(t_full[zoom_left:zoom_right], full_truth[zoom_left:zoom_right], color="#2e8b57", linewidth=1.2)
    inset.plot(
        t_full[zoom_left:zoom_right],
        full_prediction[zoom_left:zoom_right],
        color="#c0392b",
        linewidth=1.2,
        linestyle="--",
    )
    inset.set_title("Transition Zoom")
    inset.grid(alpha=0.2)
    inset.tick_params(labelsize=8)

    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def save_error_histogram(
    per_sample_errors: np.ndarray,
    output_path: str | Path,
    title: str,
    bins: int,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(10, 6))
    axis.hist(per_sample_errors, bins=bins, color="#73a9ff", edgecolor="#1f2d3d", alpha=0.8)
    axis.axvline(per_sample_errors.mean(), color="#c0392b", linestyle="--", linewidth=2, label="Mean MAE")
    axis.set_title(title)
    axis.set_xlabel("Per-sample MAE (rad)")
    axis.set_ylabel("Count")
    axis.grid(alpha=0.2)
    axis.legend()

    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)
