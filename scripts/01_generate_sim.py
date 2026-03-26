from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.simulation import generate_synthetic_pendulum_dataframe
from src.utils.config import load_yaml_config, resolve_project_path
from src.utils.runtime import create_run_dir, save_json, save_run_metadata, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic nonlinear pendulum sequences with RK4.")
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
    run_dir = create_run_dir(output_root, "01_generate_sim", config["experiment"]["name"])
    save_run_metadata(config, run_dir / "config_snapshot.yaml", run_dir)

    frame, metadata = generate_synthetic_pendulum_dataframe(config["simulation"], config["data"])
    output_csv = resolve_project_path(PROJECT_ROOT, config["simulation"]["output_csv"])
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)

    metadata["output_csv"] = str(output_csv.resolve())
    save_json(metadata, run_dir / "generation_summary.json")
    print(f"Synthetic data saved to: {output_csv}")
    print(f"Run artifacts saved to: {run_dir}")


if __name__ == "__main__":
    main()
