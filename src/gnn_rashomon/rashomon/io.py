from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def discover_runs(
    metrics_dir: Path,
    dataset: str,
    architecture: str,
    max_epochs: int | None = None,
    rashomon_type: str | None = None,
) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for path in sorted(metrics_dir.glob(f"{dataset}-{architecture}-*.json")):
        record = read_json(path)
        hyperparameters = record.get("hyperparameters", {})
        if record.get("dataset") != dataset:
            continue
        if record.get("architecture") != architecture:
            continue
        if rashomon_type is not None and record.get("rashomon_type") != rashomon_type:
            continue
        if max_epochs is not None and int(hyperparameters.get("max_epochs", -1)) != max_epochs:
            continue
        record["_record_path"] = str(path)
        runs.append(record)
    return runs
