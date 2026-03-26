from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Expected YAML mapping at {config_path}, got {type(data)!r}.")
    return data


def save_yaml_config(data: dict[str, Any], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)


def resolve_project_path(project_root: str | Path, raw_path: str | Path | None) -> Path | None:
    if raw_path is None:
        return None

    path = Path(raw_path)
    if path.is_absolute():
        return path
    return Path(project_root) / path
