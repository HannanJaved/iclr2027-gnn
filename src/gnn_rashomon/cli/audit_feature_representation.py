from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gnn_rashomon.analysis.feature_representation_audit import (
    classify_feature_representation,
)
from gnn_rashomon.cli.config import load_config
from gnn_rashomon.rashomon.io import read_json
from gnn_rashomon.training.checkpointing import checkpoint_logits

PLANETOID_NAMES = {
    "cora": "Cora",
    "citeseer": "CiteSeer",
    "pubmed": "PubMed",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether an archived Planetoid checkpoint reproduces its predictions with "
            "raw cached or canonically normalized node features."
        )
    )
    parser.add_argument("--rashomon-set", action="append", required=True)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _resolve(path: str, project_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root / candidate


def _raw_planetoid(dataset: str, project_root: Path) -> object:
    import torch
    from torch_geometric.data import Data

    if dataset not in PLANETOID_NAMES:
        raise ValueError(f"Feature-representation audit supports Planetoid only, got {dataset}.")
    config = load_config([f"dataset={dataset}"], project_root / "configs")
    root = project_root / str(config["dataset_config"]["root"])
    path = root / PLANETOID_NAMES[dataset] / "processed" / "data.pt"
    payload = torch.load(path, map_location="cpu", weights_only=False)
    data = payload[0] if isinstance(payload, tuple) else payload
    if isinstance(data, dict):
        data = Data.from_dict(data)
    return data


def _selected_run(rashomon: dict[str, object]) -> tuple[str, dict[str, object]]:
    retained = [str(value) for value in rashomon["retained_run_ids"]]
    if not retained:
        raise ValueError(f"Rashomon set {rashomon['set_id']} has no retained runs.")
    baseline = str(rashomon.get("baseline_run_id") or retained[0])
    run_id = baseline if baseline in retained else retained[0]
    runs = rashomon["runs"]
    assert isinstance(runs, dict)
    run = runs[run_id]
    assert isinstance(run, dict)
    return run_id, run


def _comparison(estimated: np.ndarray, archived: np.ndarray) -> dict[str, float]:
    absolute_error = np.abs(estimated - archived)
    return {
        "max_abs_error": float(absolute_error.max()),
        "mean_abs_error": float(absolute_error.mean()),
        "class_agreement": float(
            (estimated.argmax(axis=-1) == archived.argmax(axis=-1)).mean()
        ),
    }


def _audit_set(
    path: Path,
    *,
    project_root: Path,
    tolerance: float,
    device: str,
) -> dict[str, object]:
    import torch
    from torch_geometric.transforms import NormalizeFeatures

    rashomon = read_json(path)
    dataset = str(rashomon["dataset"])
    run_id, run = _selected_run(rashomon)
    record = read_json(_resolve(str(run["record_path"]), project_root))
    checkpoint_path = _resolve(str(record["checkpoint_path"]), project_root)
    probability_path = _resolve(str(run["probability_path"]), project_root)
    archived = (
        torch.load(probability_path, map_location="cpu", weights_only=True)
        .detach()
        .cpu()
        .numpy()
    )
    raw = _raw_planetoid(dataset, project_root)
    normalized = NormalizeFeatures()(raw.clone())
    out_channels = int(raw.y.max().item()) + 1

    estimates: dict[str, np.ndarray] = {}
    for representation, data in (("raw", raw), ("normalized", normalized)):
        logits = checkpoint_logits(
            checkpoint_path,
            x=data.x.to(device),
            edge_index=data.edge_index.to(device),
            out_channels=out_channels,
            map_location=device,
        )
        estimates[representation] = (
            torch.softmax(logits, dim=-1).detach().cpu().numpy()
        )
    raw_comparison = _comparison(estimates["raw"], archived)
    normalized_comparison = _comparison(estimates["normalized"], archived)
    classification = classify_feature_representation(
        raw_comparison["max_abs_error"],
        normalized_comparison["max_abs_error"],
        tolerance=tolerance,
    )
    return {
        "set_id": rashomon["set_id"],
        "dataset": dataset,
        "architecture": rashomon["architecture"],
        "run_id": run_id,
        "selection_rule": "baseline run when retained, otherwise first retained run",
        "checkpoint_path": str(checkpoint_path),
        "probability_path": str(probability_path),
        "tolerance": tolerance,
        "classification": classification,
        "raw_reproduces": raw_comparison["max_abs_error"] <= tolerance,
        "normalized_reproduces": normalized_comparison["max_abs_error"] <= tolerance,
        **{f"raw_{key}": value for key, value in raw_comparison.items()},
        **{f"normalized_{key}": value for key, value in normalized_comparison.items()},
    }


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]
    rows = [
        _audit_set(
            Path(raw_path),
            project_root=project_root,
            tolerance=args.tolerance,
            device=args.device,
        )
        for raw_path in args.rashomon_set
    ]
    table = pd.DataFrame(rows).sort_values(["dataset", "architecture", "set_id"])
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    table_path = Path(f"{prefix}.csv")
    summary_path = Path(f"{prefix}.summary.json")
    table.to_csv(table_path, index=False)
    counts = {str(key): int(value) for key, value in table["classification"].value_counts().items()}
    summary = {
        "audit_unit": "one selected checkpoint per Rashomon set",
        "selection_rule": "baseline run when retained, otherwise first retained run",
        "tolerance": args.tolerance,
        "set_count": len(table),
        "classification_counts": counts,
        "rashomon_sets": list(args.rashomon_set),
        "table": str(table_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(table.to_string(index=False))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
