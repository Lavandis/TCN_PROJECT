from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.amp
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from .utils.metrics import compute_evaluation_summary
from .utils.runtime import save_json


EpochCallback = Callable[[dict[str, Any]], None]


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        learning_rate: float,
        weight_decay: float = 0.0,
        amp: bool = False,
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.amp_enabled = bool(amp and device.type == "cuda")
        self.scaler = torch.amp.GradScaler(device="cuda", enabled=self.amp_enabled)

    def fit(
        self,
        train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        val_loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        epochs: int,
        output_scale: float,
        run_dir: Path | None = None,
        early_stopping_patience: int | None = None,
        save_checkpoints: bool = True,
        epoch_callback: EpochCallback | None = None,
    ) -> dict[str, Any]:
        history: list[dict[str, Any]] = []
        best_record: dict[str, Any] | None = None
        best_val_loss = float("inf")
        stale_epochs = 0

        best_checkpoint_path = run_dir / "best_model.pt" if run_dir is not None else None
        last_checkpoint_path = run_dir / "last_model.pt" if run_dir is not None else None

        for epoch in range(1, epochs + 1):
            train_loss = self._run_training_epoch(train_loader)
            val_result = self.evaluate_loader(val_loader, output_scale=output_scale, collect_inputs=False)

            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_result["loss"],
                "val_mae": val_result["metrics"]["overall_mae"],
                "val_mse": val_result["metrics"]["overall_mse"],
                "val_rmse": val_result["metrics"]["overall_rmse"],
            }
            history.append(record)

            if record["val_loss"] < best_val_loss:
                best_val_loss = record["val_loss"]
                best_record = dict(record)
                stale_epochs = 0
                if save_checkpoints and best_checkpoint_path is not None:
                    self._save_checkpoint(best_checkpoint_path, epoch=epoch, best_val_loss=best_val_loss)
            else:
                stale_epochs += 1

            if epoch_callback is not None:
                epoch_callback(record)

            if early_stopping_patience is not None and stale_epochs >= early_stopping_patience:
                break

        if save_checkpoints and last_checkpoint_path is not None:
            self._save_checkpoint(last_checkpoint_path, epoch=history[-1]["epoch"], best_val_loss=best_val_loss)

        result = {
            "history": history,
            "best_epoch": None if best_record is None else best_record["epoch"],
            "best_val_loss": best_val_loss,
            "best_metrics": None if best_record is None else best_record,
            "best_checkpoint": None if best_checkpoint_path is None else str(best_checkpoint_path),
            "last_checkpoint": None if last_checkpoint_path is None else str(last_checkpoint_path),
        }

        if run_dir is not None:
            history_path = run_dir / "train_history.csv"
            pd.DataFrame(history).to_csv(history_path, index=False)
            save_json(
                {
                    "best_epoch": result["best_epoch"],
                    "best_val_loss": result["best_val_loss"],
                    "best_checkpoint": result["best_checkpoint"],
                    "last_checkpoint": result["last_checkpoint"],
                    "epochs_ran": history[-1]["epoch"],
                },
                run_dir / "fit_summary.json",
            )

        return result

    def evaluate_loader(
        self,
        loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
        output_scale: float,
        collect_inputs: bool,
    ) -> dict[str, Any]:
        self.model.eval()
        losses: list[float] = []
        collected_inputs: list[np.ndarray] = []
        collected_targets: list[np.ndarray] = []
        collected_predictions: list[np.ndarray] = []

        with torch.no_grad():
            for inputs, targets in loader:
                inputs = inputs.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)

                predictions = self.model(inputs)
                loss = self.criterion(predictions, targets)
                losses.append(float(loss.item()))

                predictions_np = predictions.detach().cpu().numpy() * output_scale
                targets_np = targets.detach().cpu().numpy() * output_scale
                collected_predictions.append(predictions_np)
                collected_targets.append(targets_np)
                if collect_inputs:
                    collected_inputs.append(inputs.detach().cpu().numpy()[:, 0, :] * output_scale)

        targets_array = np.concatenate(collected_targets, axis=0)
        predictions_array = np.concatenate(collected_predictions, axis=0)
        metrics, per_sample = compute_evaluation_summary(targets_array, predictions_array)

        result = {
            "loss": float(np.mean(losses)),
            "metrics": metrics,
            "per_sample": per_sample,
            "targets": targets_array,
            "predictions": predictions_array,
            "inputs": None if not collect_inputs else np.concatenate(collected_inputs, axis=0),
        }
        return result

    def _run_training_epoch(
        self,
        loader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    ) -> float:
        self.model.train()
        batch_losses: list[float] = []

        for inputs, targets in loader:
            inputs = inputs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
                predictions = self.model(inputs)
                loss = self.criterion(predictions, targets)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            batch_losses.append(float(loss.item()))

        return float(np.mean(batch_losses))

    def _save_checkpoint(self, output_path: Path, epoch: int, best_val_loss: float) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            output_path,
        )


def load_model_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        state_dict = payload["model_state_dict"]
    elif isinstance(payload, dict):
        state_dict = payload
    else:
        raise TypeError(f"Unsupported checkpoint payload type: {type(payload)!r}")

    model.load_state_dict(state_dict)
    return payload if isinstance(payload, dict) else {}
