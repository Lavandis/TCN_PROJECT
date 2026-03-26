from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import optuna
import pandas as pd

from src.dataset import (
    build_dataloader,
    build_window_spec,
    load_sequence_matrix,
    select_rows,
    validate_sequence_matrix,
)
from src.models.tcn import build_tcn_from_config
from src.trainer import Trainer
from src.utils.config import load_yaml_config, resolve_project_path
from src.utils.runtime import create_run_dir, prepare_split_indices, save_json, save_run_metadata, seed_everything, select_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter search on synthetic pendulum data.")
    parser.add_argument("--config", required=True, help="Path to the pretrain YAML config.")
    return parser.parse_args()


def sample_trial_params(trial: optuna.Trial, search_space: dict[str, Any]) -> dict[str, Any]:
    lr_space = search_space["learning_rate"]
    dropout_space = search_space["dropout"]
    return {
        "kernel_size": trial.suggest_categorical("kernel_size", search_space["kernel_size"]),
        "num_layers": trial.suggest_categorical("num_layers", search_space["num_layers"]),
        "channel_width": trial.suggest_categorical("channel_width", search_space["channel_width"]),
        "dropout": trial.suggest_float("dropout", float(dropout_space["low"]), float(dropout_space["high"])),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            float(lr_space["low"]),
            float(lr_space["high"]),
            log=bool(lr_space.get("log", False)),
        ),
        "batch_size": trial.suggest_categorical("batch_size", search_space["batch_size"]),
    }


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(PROJECT_ROOT, args.config)
    if config_path is None:
        raise ValueError("A config path is required.")

    config = load_yaml_config(config_path)
    seed_everything(int(config["experiment"]["seed"]))

    output_root = resolve_project_path(PROJECT_ROOT, config["artifacts"]["output_root"])
    run_dir = create_run_dir(output_root, "02_tune_optuna", config["experiment"]["name"])
    save_run_metadata(config, run_dir / "config_snapshot.yaml", run_dir)

    data_config = config["data"]
    train_config = config["train"]
    optuna_config = config["optuna"]
    window_spec = build_window_spec(data_config)
    device = select_device(config.get("device", "auto"))

    synthetic_csv = resolve_project_path(PROJECT_ROOT, data_config["synthetic_csv"])
    if not synthetic_csv.exists():
        raise FileNotFoundError(f"Synthetic CSV not found: {synthetic_csv}")

    matrix, _ = load_sequence_matrix(str(synthetic_csv), has_sample_id_column=bool(data_config.get("has_sample_id_column")))
    validate_sequence_matrix(matrix, window_spec, str(synthetic_csv))

    split_payload = prepare_split_indices(len(matrix), data_config["split"], PROJECT_ROOT, run_dir)
    indices = split_payload["indices"]
    train_sequences = select_rows(matrix, indices["train"])
    val_sequences = select_rows(matrix, indices["val"])
    scale_factor = float(data_config.get("scale_factor", 1.0))

    def objective(trial: optuna.Trial) -> float:
        trial_params = sample_trial_params(trial, optuna_config["search_space"])

        model_config = dict(config["model"])
        model_config.update(
            {
                "kernel_size": trial_params["kernel_size"],
                "num_layers": trial_params["num_layers"],
                "channel_width": trial_params["channel_width"],
                "dropout": trial_params["dropout"],
            }
        )

        batch_size = int(trial_params["batch_size"])
        train_loader = build_dataloader(
            sequences=train_sequences,
            window_spec=window_spec,
            scale_factor=scale_factor,
            batch_size=batch_size,
            shuffle=True,
            num_workers=int(train_config["num_workers"]),
            pin_memory=bool(train_config["pin_memory"]),
        )
        val_loader = build_dataloader(
            sequences=val_sequences,
            window_spec=window_spec,
            scale_factor=scale_factor,
            batch_size=batch_size,
            shuffle=False,
            num_workers=int(train_config["num_workers"]),
            pin_memory=bool(train_config["pin_memory"]),
        )

        model = build_tcn_from_config(model_config, output_steps=window_spec.pred_steps)
        trainer = Trainer(
            model=model,
            device=device,
            learning_rate=float(trial_params["learning_rate"]),
            weight_decay=float(train_config["weight_decay"]),
            amp=bool(train_config["amp"]),
        )

        def report_epoch(epoch_record: dict[str, Any]) -> None:
            trial.report(float(epoch_record["val_loss"]), step=int(epoch_record["epoch"]))
            if trial.should_prune():
                raise optuna.TrialPruned()

        fit_result = trainer.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=int(optuna_config["train_epochs_per_trial"]),
            output_scale=scale_factor,
            run_dir=None,
            early_stopping_patience=None,
            save_checkpoints=False,
            epoch_callback=report_epoch,
        )
        best_metrics = fit_result["best_metrics"] or {}
        trial.set_user_attr("val_mae", best_metrics.get("val_mae"))
        return float(fit_result["best_val_loss"])

    pruner = optuna.pruners.MedianPruner(n_warmup_steps=int(optuna_config["pruner_warmup_steps"]))
    study = optuna.create_study(direction="minimize", study_name=optuna_config["study_name"], pruner=pruner)
    study.optimize(
        objective,
        n_trials=int(optuna_config["n_trials"]),
        timeout=None if optuna_config.get("timeout_seconds") is None else int(optuna_config["timeout_seconds"]),
    )

    best_params = study.best_params
    best_trial = study.best_trial
    recommended_model = dict(config["model"])
    recommended_model.update(
        {
            "kernel_size": best_params["kernel_size"],
            "num_layers": best_params["num_layers"],
            "channel_width": best_params["channel_width"],
            "dropout": best_params["dropout"],
        }
    )
    recommended_train = dict(train_config)
    recommended_train["learning_rate"] = best_params["learning_rate"]
    recommended_train["batch_size"] = best_params["batch_size"]

    study.trials_dataframe().to_csv(run_dir / "study_trials.csv", index=False)
    save_json(best_params, run_dir / "best_params.json")
    save_json(
        {
            "best_value": float(study.best_value),
            "best_trial_number": int(best_trial.number),
            "recommended_model": recommended_model,
            "recommended_train": recommended_train,
        },
        run_dir / "optuna_summary.json",
    )

    print(f"Best parameters saved to: {run_dir / 'best_params.json'}")
    print(pd.Series(best_params).to_string())


if __name__ == "__main__":
    main()
