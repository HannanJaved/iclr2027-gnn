from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def parse_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def set_nested(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    current = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def load_config(argv: list[str], config_dir: Path) -> dict[str, Any]:
    config = load_yaml(config_dir / "base.yaml")
    overrides: dict[str, Any] = {}
    for arg in argv:
        if "=" not in arg:
            raise ValueError(f"Expected override in key=value form, got {arg!r}")
        key, raw_value = arg.split("=", 1)
        set_nested(overrides, key, parse_value(raw_value))

    dataset = str(overrides.get("dataset", config.get("dataset", "cora")))
    model = str(overrides.get("model", config.get("model", "gcn")))
    deep_update(config, {"dataset_config": load_yaml(config_dir / "datasets" / f"{dataset}.yaml")})
    deep_update(config, {"model_config": load_yaml(config_dir / "models" / f"{model}.yaml")})
    deep_update(config, overrides)
    return config
