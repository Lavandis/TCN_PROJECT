from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class WindowSpec:
    fps: int
    input_seconds: float
    pred_seconds: float

    @property
    def input_steps(self) -> int:
        return int(round(self.fps * self.input_seconds))

    @property
    def pred_steps(self) -> int:
        return int(round(self.fps * self.pred_seconds))

    @property
    def total_steps(self) -> int:
        return self.input_steps + self.pred_steps


def build_window_spec(data_config: dict[str, Any]) -> WindowSpec:
    return WindowSpec(
        fps=int(data_config["fps"]),
        input_seconds=float(data_config["input_seconds"]),
        pred_seconds=float(data_config["pred_seconds"]),
    )


def load_sequence_matrix(
    csv_path: str,
    has_sample_id_column: bool = False,
) -> tuple[np.ndarray, list[str]]:
    frame = pd.read_csv(csv_path)
    if frame.empty:
        raise ValueError(f"Sequence CSV is empty: {csv_path}")

    if has_sample_id_column:
        sample_ids = frame.iloc[:, 0].astype(str).tolist()
        values = frame.iloc[:, 1:]
    else:
        sample_ids = [str(index) for index in range(len(frame))]
        values = frame

    matrix = values.to_numpy(dtype=np.float32, copy=True)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D matrix from {csv_path}, got shape {matrix.shape}.")

    return matrix, sample_ids


def validate_sequence_matrix(matrix: np.ndarray, window_spec: WindowSpec, csv_path: str) -> None:
    if matrix.shape[1] < window_spec.total_steps:
        raise ValueError(
            "Sequence length is shorter than required window length: "
            f"{csv_path} has {matrix.shape[1]} columns, but needs at least {window_spec.total_steps}."
        )


def create_split_indices(
    num_samples: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[int]]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if not np.isclose(ratio_sum, 1.0, atol=1e-6):
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum:.6f}.")

    if num_samples < 3:
        raise ValueError("Need at least 3 samples to create train/val/test splits.")

    permutation = np.random.default_rng(seed).permutation(num_samples)

    train_end = int(num_samples * train_ratio)
    val_end = train_end + int(num_samples * val_ratio)

    train_indices = permutation[:train_end].tolist()
    val_indices = permutation[train_end:val_end].tolist()
    test_indices = permutation[val_end:].tolist()

    if not train_indices or not val_indices or not test_indices:
        raise ValueError(
            "Split configuration produced an empty partition. "
            f"Counts: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}."
        )

    return {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }


def select_rows(matrix: np.ndarray, indices: list[int]) -> np.ndarray:
    return matrix[np.asarray(indices, dtype=np.int64)]


def select_sample_ids(sample_ids: list[str], indices: list[int]) -> list[str]:
    return [sample_ids[index] for index in indices]


class SequenceDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        sequences: np.ndarray,
        window_spec: WindowSpec,
        scale_factor: float = 1.0,
    ) -> None:
        if scale_factor == 0:
            raise ValueError("scale_factor must be non-zero.")

        self.sequences = np.asarray(sequences, dtype=np.float32)
        self.window_spec = window_spec
        self.scale_factor = float(scale_factor)

        if self.sequences.ndim != 2:
            raise ValueError(f"Expected sequences to be 2D, got shape {self.sequences.shape}.")

        if self.sequences.shape[1] < window_spec.total_steps:
            raise ValueError(
                f"Sequence width {self.sequences.shape[1]} is smaller than required {window_spec.total_steps}."
            )

    def __len__(self) -> int:
        return int(self.sequences.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.sequences[index, : self.window_spec.total_steps] / self.scale_factor
        source = row[: self.window_spec.input_steps]
        target = row[self.window_spec.input_steps : self.window_spec.total_steps]

        source_tensor = torch.from_numpy(source.astype(np.float32, copy=False)).unsqueeze(0)
        target_tensor = torch.from_numpy(target.astype(np.float32, copy=False))
        return source_tensor, target_tensor


def build_dataloader(
    sequences: np.ndarray,
    window_spec: WindowSpec,
    scale_factor: float,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    dataset = SequenceDataset(
        sequences=sequences,
        window_spec=window_spec,
        scale_factor=scale_factor,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
