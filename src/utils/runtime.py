from __future__ import annotations

import json
import random
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from ..dataset import create_split_indices
from .config import resolve_project_path, save_yaml_config


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_name: str) -> torch.device:
    requested = device_name.lower()
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(requested)


def create_run_dir(output_root: str | Path, stage_name: str, experiment_name: str) -> Path:
    root = Path(output_root)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    slug = _slugify(experiment_name)
    stage_root = root / stage_name
    stage_root.mkdir(parents=True, exist_ok=True)

    run_dir = stage_root / f"{timestamp}_{slug}"
    collision_index = 1
    while run_dir.exists():
        run_dir = stage_root / f"{timestamp}_{slug}_{collision_index:02d}"
        collision_index += 1
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def save_json(payload: dict[str, Any], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)


def save_run_metadata(
    config: dict[str, Any],
    config_snapshot_path: str | Path,
    run_dir: str | Path,
) -> None:
    save_yaml_config(config, config_snapshot_path)
    save_json(
        {
            "run_dir": str(Path(run_dir).resolve()),
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
        Path(run_dir) / "run_metadata.json",
    )


def prepare_split_indices(
    num_samples: int,
    split_config: dict[str, Any],
    project_root: str | Path,
    run_dir: str | Path,
) -> dict[str, Any]:
    persist_path = resolve_project_path(project_root, split_config.get("persist_path"))
    ratios = {
        "train": float(split_config["train"]),
        "val": float(split_config["val"]),
        "test": float(split_config["test"]),
    }
    seed = int(split_config["seed"])

    if persist_path is not None and persist_path.exists():
        with persist_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        indices = create_split_indices(
            num_samples=num_samples,
            train_ratio=ratios["train"],
            val_ratio=ratios["val"],
            test_ratio=ratios["test"],
            seed=seed,
        )
        payload = {
            "num_samples": num_samples,
            "ratios": ratios,
            "seed": seed,
            "indices": indices,
        }
        if persist_path is not None:
            save_json(payload, persist_path)

    if int(payload["num_samples"]) != num_samples:
        raise ValueError(
            "Persisted split file does not match current dataset size: "
            f"expected {num_samples}, found {payload['num_samples']}."
        )

    save_json(payload, Path(run_dir) / "split_indices.json")
    return payload


def maybe_publish_checkpoint_alias(
    checkpoint_path: str | Path,
    alias_dir: str | Path | None,
    alias_name: str | None,
) -> Path | None:
    if alias_dir is None or alias_name is None:
        return None

    source = Path(checkpoint_path)
    alias_path = Path(alias_dir) / alias_name
    alias_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, alias_path)
    save_json(
        {
            "source_checkpoint": str(source.resolve()),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        },
        alias_path.with_suffix(alias_path.suffix + ".source.json"),
    )
    return alias_path


def _slugify(text: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return normalized or "run"


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value)!r} is not JSON serializable.")
