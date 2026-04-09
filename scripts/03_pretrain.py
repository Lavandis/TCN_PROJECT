from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import (
    build_dataloader,
    build_window_spec,
    load_sequence_matrix,
    select_rows,
    select_sample_ids,
    validate_sequence_matrix,
)
from src.models.tcn import build_tcn_from_config
from src.trainer import Trainer, load_model_checkpoint
from src.utils.config import load_yaml_config, resolve_project_path
from src.utils.plot import choose_sample_index, save_error_histogram, save_trajectory_plot
from src.utils.runtime import (
    create_run_dir,
    maybe_publish_checkpoint_alias,
    prepare_split_indices,
    save_json,
    save_run_metadata,
    seed_everything,
    select_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain the TCN on synthetic pendulum data.")
    parser.add_argument("--config", required=True, help="Path to the pretrain YAML config.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_project_path(PROJECT_ROOT, args.config)
    if config_path is None:
        raise ValueError("A config path is required.")

    config = load_yaml_config(config_path)
    seed_everything(int(config["experiment"]["seed"]))

    output_root = resolve_project_path(PROJECT_ROOT, config["artifacts"]["output_root"])
    alias_dir = resolve_project_path(PROJECT_ROOT, config["artifacts"].get("alias_dir"))
    run_dir = create_run_dir(output_root, "03_pretrain", config["experiment"]["name"])
    save_run_metadata(config, run_dir / "config_snapshot.yaml", run_dir)

    data_config = config["data"]
    train_config = config["train"]
    plot_config = config["plot"]
    window_spec = build_window_spec(data_config)
    device = select_device(config.get("device", "auto"))
    scale_factor = float(data_config.get("scale_factor", 1.0))

    synthetic_csv = resolve_project_path(PROJECT_ROOT, data_config["synthetic_csv"])
    if not synthetic_csv.exists():
        raise FileNotFoundError(f"Synthetic CSV not found: {synthetic_csv}")

    matrix, sample_ids = load_sequence_matrix(str(synthetic_csv), has_sample_id_column=bool(data_config.get("has_sample_id_column")))
    validate_sequence_matrix(matrix, window_spec, str(synthetic_csv))

    split_payload = prepare_split_indices(len(matrix), data_config["split"], PROJECT_ROOT, run_dir)
    indices = split_payload["indices"]

    train_loader = build_dataloader(
        sequences=select_rows(matrix, indices["train"]),
        window_spec=window_spec,
        scale_factor=scale_factor,
        batch_size=int(train_config["batch_size"]),
        shuffle=True,
        num_workers=int(train_config["num_workers"]),
        pin_memory=bool(train_config["pin_memory"]),
    )
    val_loader = build_dataloader(
        sequences=select_rows(matrix, indices["val"]),
        window_spec=window_spec,
        scale_factor=scale_factor,
        batch_size=int(train_config["batch_size"]),
        shuffle=False,
        num_workers=int(train_config["num_workers"]),
        pin_memory=bool(train_config["pin_memory"]),
    )
    test_loader = build_dataloader(
        sequences=select_rows(matrix, indices["test"]),
        window_spec=window_spec,
        scale_factor=scale_factor,
        batch_size=int(train_config["batch_size"]),
        shuffle=False,
        num_workers=int(train_config["num_workers"]),
        pin_memory=bool(train_config["pin_memory"]),
    )

    model = build_tcn_from_config(config["model"], output_steps=window_spec.pred_steps)
    trainer = Trainer(
        model=model,
        device=device,
        learning_rate=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
        amp=bool(train_config["amp"]),
    )
    fit_result = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=int(train_config["epochs"]),
        output_scale=scale_factor,
        run_dir=run_dir,
        early_stopping_patience=int(train_config["early_stopping_patience"]),
        save_checkpoints=True,
        show_progress=True,
    )

    best_checkpoint = fit_result["best_checkpoint"]
    if best_checkpoint is None:
        raise RuntimeError("Pretraining did not produce a best checkpoint.")
    load_model_checkpoint(model, best_checkpoint, device)

    evaluation = trainer.evaluate_loader(
        test_loader,
        output_scale=scale_factor,
        collect_inputs=True,
        show_progress=True,
        progress_desc="Test",
    )
    save_json(evaluation["metrics"], run_dir / "test_metrics.json")

    test_sample_ids = select_sample_ids(sample_ids, indices["test"])
    chosen_index = choose_sample_index(
        num_samples=len(test_sample_ids),
        sample_index=plot_config.get("sample_index"),
        random_seed=int(plot_config["random_seed"]),
    )

    save_trajectory_plot(
        input_sequence=evaluation["inputs"][chosen_index],
        target_sequence=evaluation["targets"][chosen_index],
        prediction_sequence=evaluation["predictions"][chosen_index],
        fps=window_spec.fps,
        output_path=run_dir / "synthetic_trajectory_plot.png",
        title=f"Synthetic Test Prediction | sample={test_sample_ids[chosen_index]}",
        zoom_history_seconds=float(plot_config["zoom_history_seconds"]),
        zoom_future_seconds=float(plot_config["zoom_future_seconds"]),
    )
    save_error_histogram(
        per_sample_errors=evaluation["per_sample"]["mae"],
        output_path=run_dir / "synthetic_error_histogram.png",
        title="Synthetic Test MAE Distribution",
        bins=int(plot_config["histogram_bins"]),
    )

    alias_path = maybe_publish_checkpoint_alias(best_checkpoint, alias_dir, "pretrained_model_latest.pt")
    save_json(
        {
            "best_checkpoint": best_checkpoint,
            "published_alias": None if alias_path is None else str(alias_path.resolve()),
        },
        run_dir / "checkpoint_alias.json",
    )

    print(f"Best checkpoint: {best_checkpoint}")
    if alias_path is not None:
        print(f"Published alias: {alias_path}")
    print(f"Test metrics saved to: {run_dir / 'test_metrics.json'}")


if __name__ == "__main__":
    main()
