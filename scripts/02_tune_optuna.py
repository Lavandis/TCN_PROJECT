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
from optuna.trial import TrialState

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


def build_trial_model_config(base_model_config: dict[str, Any], trial_params: dict[str, Any]) -> dict[str, Any]:
    model_config = dict(base_model_config)
    model_config.update(
        {
            "kernel_size": int(trial_params["kernel_size"]),
            "num_layers": int(trial_params["num_layers"]),
            "channel_width": int(trial_params["channel_width"]),
            "dropout": float(trial_params["dropout"]),
        }
    )
    return model_config


def extract_objective_value(metrics: dict[str, Any], metric_name: str) -> float:
    if metric_name not in metrics:
        available = ", ".join(sorted(metrics.keys()))
        raise KeyError(f"Objective metric '{metric_name}' not found. Available metrics: {available}")
    return float(metrics[metric_name])


def train_with_params(
    *,
    base_seed: int,
    run_seed_offset: int,
    trial_params: dict[str, Any],
    base_model_config: dict[str, Any],
    train_config: dict[str, Any],
    optuna_config: dict[str, Any],
    train_sequences,
    val_sequences,
    window_spec,
    scale_factor: float,
    device,
    objective_metric: str,
) -> dict[str, Any]:
    seed_everything(base_seed + run_seed_offset)

    model_config = build_trial_model_config(base_model_config, trial_params)
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

    fit_result = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=int(optuna_config["active_epochs"]),
        output_scale=scale_factor,
        run_dir=None,
        early_stopping_patience=optuna_config.get("active_early_stopping_patience"),
        save_checkpoints=False,
        epoch_callback=optuna_config.get("epoch_callback"),
    )
    best_metrics = fit_result["best_metrics"] or {}
    objective_value = extract_objective_value(best_metrics, objective_metric)
    return {
        "fit_result": fit_result,
        "best_metrics": best_metrics,
        "objective_value": objective_value,
        "model_config": model_config,
    }


def refine_top_trials(
    *,
    study: optuna.Study,
    run_dir: Path,
    base_seed: int,
    base_model_config: dict[str, Any],
    train_config: dict[str, Any],
    optuna_config: dict[str, Any],
    train_sequences,
    val_sequences,
    window_spec,
    scale_factor: float,
    device,
    objective_metric: str,
) -> dict[str, Any] | None:
    refine_top_k = int(optuna_config.get("refine_top_k", 0) or 0)
    refine_epochs = int(optuna_config.get("refine_epochs", 0) or 0)
    if refine_top_k <= 0 or refine_epochs <= 0:
        return None

    completed_trials = [trial for trial in study.trials if trial.state == TrialState.COMPLETE]
    if not completed_trials:
        return None

    top_trials = sorted(completed_trials, key=lambda item: float(item.value))[:refine_top_k]
    refinement_rows: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None

    for refine_rank, trial in enumerate(top_trials, start=1):
        trial_params = {
            "kernel_size": int(trial.params["kernel_size"]),
            "num_layers": int(trial.params["num_layers"]),
            "channel_width": int(trial.params["channel_width"]),
            "dropout": float(trial.params["dropout"]),
            "learning_rate": float(trial.params["learning_rate"]),
            "batch_size": int(trial.params["batch_size"]),
        }
        refinement_config = dict(optuna_config)
        refinement_config["active_epochs"] = refine_epochs
        refinement_config["active_early_stopping_patience"] = optuna_config.get("refine_early_stopping_patience")
        refinement_config["epoch_callback"] = None
        training_result = train_with_params(
            base_seed=base_seed,
            run_seed_offset=1000 + refine_rank,
            trial_params=trial_params,
            base_model_config=base_model_config,
            train_config=train_config,
            optuna_config=refinement_config,
            train_sequences=train_sequences,
            val_sequences=val_sequences,
            window_spec=window_spec,
            scale_factor=scale_factor,
            device=device,
            objective_metric=objective_metric,
        )

        row = {
            "refine_rank": refine_rank,
            "source_trial_number": int(trial.number),
            "search_objective_value": float(trial.value),
            "refined_objective_value": float(training_result["objective_value"]),
            "best_epoch": training_result["fit_result"]["best_epoch"],
            "kernel_size": trial_params["kernel_size"],
            "num_layers": trial_params["num_layers"],
            "channel_width": trial_params["channel_width"],
            "dropout": trial_params["dropout"],
            "learning_rate": trial_params["learning_rate"],
            "batch_size": trial_params["batch_size"],
            "val_loss": training_result["best_metrics"].get("val_loss"),
            "val_mae": training_result["best_metrics"].get("val_mae"),
            "val_rmse": training_result["best_metrics"].get("val_rmse"),
        }
        refinement_rows.append(row)

        if best_result is None or row["refined_objective_value"] < best_result["refined_objective_value"]:
            best_result = row

    pd.DataFrame(refinement_rows).to_csv(run_dir / "refinement_trials.csv", index=False)
    save_json(refinement_rows, run_dir / "refinement_trials.json")
    return best_result


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
    base_seed = int(config["experiment"]["seed"])
    objective_metric = str(optuna_config.get("objective_metric", "val_loss"))

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

        def report_epoch(epoch_record: dict[str, Any]) -> None:
            trial.report(extract_objective_value(epoch_record, objective_metric), step=int(epoch_record["epoch"]))
            if trial.should_prune():
                raise optuna.TrialPruned()

        training_optuna_config = dict(optuna_config)
        training_optuna_config["active_epochs"] = int(optuna_config["train_epochs_per_trial"])
        training_optuna_config["active_early_stopping_patience"] = optuna_config.get("trial_early_stopping_patience")
        training_optuna_config["epoch_callback"] = report_epoch
        training_result = train_with_params(
            base_seed=base_seed,
            run_seed_offset=int(trial.number),
            trial_params=trial_params,
            base_model_config=config["model"],
            train_config=train_config,
            optuna_config=training_optuna_config,
            train_sequences=train_sequences,
            val_sequences=val_sequences,
            window_spec=window_spec,
            scale_factor=scale_factor,
            device=device,
            objective_metric=objective_metric,
        )
        best_metrics = training_result["best_metrics"]
        trial.set_user_attr("val_loss", best_metrics.get("val_loss"))
        trial.set_user_attr("val_mae", best_metrics.get("val_mae"))
        trial.set_user_attr("val_rmse", best_metrics.get("val_rmse"))
        return float(training_result["objective_value"])

    sampler = optuna.samplers.TPESampler(seed=base_seed)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=int(optuna_config["pruner_warmup_steps"]))
    study = optuna.create_study(
        direction="minimize",
        study_name=optuna_config["study_name"],
        pruner=pruner,
        sampler=sampler,
    )
    study.optimize(
        objective,
        n_trials=int(optuna_config["n_trials"]),
        timeout=None if optuna_config.get("timeout_seconds") is None else int(optuna_config["timeout_seconds"]),
    )

    study_best_params = dict(study.best_params)
    study_best_trial = study.best_trial
    refinement_best = refine_top_trials(
        study=study,
        run_dir=run_dir,
        base_seed=base_seed,
        base_model_config=config["model"],
        train_config=train_config,
        optuna_config=optuna_config,
        train_sequences=train_sequences,
        val_sequences=val_sequences,
        window_spec=window_spec,
        scale_factor=scale_factor,
        device=device,
        objective_metric=objective_metric,
    )

    if refinement_best is None:
        best_params = study_best_params
        selected_source = {
            "selection_stage": "search",
            "trial_number": int(study_best_trial.number),
            "objective_metric": objective_metric,
            "objective_value": float(study.best_value),
        }
    else:
        best_params = {
            "kernel_size": int(refinement_best["kernel_size"]),
            "num_layers": int(refinement_best["num_layers"]),
            "channel_width": int(refinement_best["channel_width"]),
            "dropout": float(refinement_best["dropout"]),
            "learning_rate": float(refinement_best["learning_rate"]),
            "batch_size": int(refinement_best["batch_size"]),
        }
        selected_source = {
            "selection_stage": "refinement",
            "source_trial_number": int(refinement_best["source_trial_number"]),
            "objective_metric": objective_metric,
            "objective_value": float(refinement_best["refined_objective_value"]),
            "search_objective_value": float(refinement_best["search_objective_value"]),
            "best_epoch": refinement_best["best_epoch"],
        }

    recommended_model = build_trial_model_config(config["model"], best_params)
    recommended_train = dict(train_config)
    recommended_train["learning_rate"] = float(best_params["learning_rate"])
    recommended_train["batch_size"] = int(best_params["batch_size"])

    study.trials_dataframe().to_csv(run_dir / "study_trials.csv", index=False)
    save_json(study_best_params, run_dir / "study_best_params.json")
    save_json(best_params, run_dir / "best_params.json")
    save_json(
        {
            "objective_metric": objective_metric,
            "study_best_value": float(study.best_value),
            "study_best_trial_number": int(study_best_trial.number),
            "study_best_params": study_best_params,
            "selected_params": best_params,
            "selected_source": selected_source,
            "recommended_model": recommended_model,
            "recommended_train": recommended_train,
        },
        run_dir / "optuna_summary.json",
    )

    print(f"Selected parameters saved to: {run_dir / 'best_params.json'}")
    print(pd.Series(best_params).to_string())


if __name__ == "__main__":
    main()
