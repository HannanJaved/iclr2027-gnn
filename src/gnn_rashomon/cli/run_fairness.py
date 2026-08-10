from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gnn_rashomon.cli.config import load_config
from gnn_rashomon.data.loaders import load_graph
from gnn_rashomon.fairness.metrics import (
    dispersion,
    fairness_metrics,
)
from gnn_rashomon.rashomon.io import read_json
from gnn_rashomon.rewiring.random_rewire import degree_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute fairness dispersion over a Rashomon set.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--rashomon-set", required=True)
    parser.add_argument("--multiplicity-table", default=None)
    parser.add_argument("--graph-path", default=None)
    parser.add_argument("--sensitive-attribute", default="sex")
    parser.add_argument("--positive-label", type=int, default=1)
    parser.add_argument("--degree-quantile", type=float, default=0.25)
    parser.add_argument(
        "--equalized-odds-combine", choices=["max", "sum", "average"], default="max"
    )
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def _to_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _load_probabilities(path: str) -> np.ndarray:
    import torch

    return torch.load(path, map_location="cpu", weights_only=True).detach().cpu().numpy()


def _metric_summary(frame: pd.DataFrame, metric: str) -> dict[str, object]:
    values = frame[metric].to_numpy(dtype=float)
    best_idx = int(np.nanargmin(values)) if np.any(~np.isnan(values)) else -1
    worst_idx = int(np.nanargmax(values)) if np.any(~np.isnan(values)) else -1
    out: dict[str, object] = dispersion(values)
    out["best_run_id"] = None if best_idx < 0 else str(frame.iloc[best_idx]["run_id"])
    out["worst_run_id"] = None if worst_idx < 0 else str(frame.iloc[worst_idx]["run_id"])
    out["validation_accuracy_correlation"] = float(
        frame[[metric, "validation_accuracy"]].corr(method="spearman").iloc[0, 1]
    )
    return out


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]
    config = load_config([f"dataset={args.dataset}"], project_root / "configs")
    graph = load_graph(
        args.dataset,
        root=str(project_root / config["dataset_config"]["root"]),
        graph_path=args.graph_path,
        dataset_config=config["dataset_config"],
    )
    data = graph.data
    sensitive_value = graph.sensitive_attributes.get(args.sensitive_attribute)
    if sensitive_value is None and hasattr(data, "sensitive_attr"):
        sensitive_value = data.sensitive_attr
    if sensitive_value is None:
        raise SystemExit(f"No sensitive attribute named {args.sensitive_attribute!r} found.")

    labels = _to_numpy(data.y).astype(int)
    sensitive = _to_numpy(sensitive_value).astype(int)
    degree = degree_sequence(_to_numpy(data.edge_index).astype(int), int(data.num_nodes))
    label_mask = _to_numpy(getattr(data, "label_mask", np.ones_like(labels, dtype=bool))).astype(
        bool
    )
    validation_mask = _to_numpy(data.val_mask).astype(bool) & label_mask & (labels >= 0)
    test_mask = _to_numpy(data.test_mask).astype(bool) & label_mask & (labels >= 0)
    if not np.any(validation_mask) or not np.any(test_mask):
        raise SystemExit(
            f"No valid labels available for fairness evaluation on dataset={args.dataset}."
        )
    rashomon = read_json(Path(args.rashomon_set))
    rows: list[dict[str, object]] = []
    for run_id in rashomon["retained_run_ids"]:
        run = rashomon["runs"][run_id]
        predicted = _load_probabilities(run["probability_path"]).argmax(axis=-1).astype(int)
        validation = fairness_metrics(
            predicted,
            labels,
            sensitive,
            degree,
            validation_mask,
            positive_label=args.positive_label,
            equalized_odds_combine=args.equalized_odds_combine,
            degree_quantile=args.degree_quantile,
        )
        test = fairness_metrics(
            predicted,
            labels,
            sensitive,
            degree,
            test_mask,
            positive_label=args.positive_label,
            equalized_odds_combine=args.equalized_odds_combine,
            degree_quantile=args.degree_quantile,
        )
        rows.append(
            {
                "run_id": run_id,
                "validation_accuracy": float(run["validation_accuracy"]),
                "test_accuracy_report_only": float(run["test_accuracy"]),
                **{f"validation_{key}": value for key, value in validation.items()},
                **{f"test_{key}_report_only": value for key, value in test.items()},
            }
        )

    output_dir = Path(args.output_dir)
    set_id = rashomon["set_id"]
    table = pd.DataFrame(rows)
    table_path = output_dir / "metrics" / f"{set_id}.fairness_models.csv"
    summary_path = output_dir / "metrics" / f"{set_id}.fairness_summary.json"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(table_path, index=False)

    summary: dict[str, object] = {
        "set_id": set_id,
        "dataset": args.dataset,
        "sensitive_attribute": args.sensitive_attribute,
        "positive_label": args.positive_label,
        "equalized_odds_combine": args.equalized_odds_combine,
        "degree_quantile": args.degree_quantile,
        "num_models": int(len(table)),
        "selection_split": "validation",
        "test_metrics_are_report_only": True,
        "validation_demographic_parity_gap": _metric_summary(
            table, "validation_demographic_parity_gap"
        ),
        "validation_equalized_odds_gap": _metric_summary(table, "validation_equalized_odds_gap"),
        "validation_degree_disparity": _metric_summary(table, "validation_degree_disparity"),
        "test_demographic_parity_gap_report_only": _metric_summary(
            table, "test_demographic_parity_gap_report_only"
        ),
        "test_equalized_odds_gap_report_only": _metric_summary(
            table, "test_equalized_odds_gap_report_only"
        ),
        "test_degree_disparity_report_only": _metric_summary(
            table, "test_degree_disparity_report_only"
        ),
    }
    if args.multiplicity_table:
        multiplicity = pd.read_csv(args.multiplicity_table)
        multiplicity["sensitive"] = sensitive
        summary["group_multiplicity_concentration"] = {
            str(group): float(subset["rashomon_capacity"].mean())
            for group, subset in multiplicity.groupby("sensitive")
        }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote fairness model table to {table_path}")
    print(f"Wrote fairness summary to {summary_path}")


if __name__ == "__main__":
    main()
