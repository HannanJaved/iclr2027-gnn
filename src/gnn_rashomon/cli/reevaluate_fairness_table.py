from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gnn_rashomon.cli.config import load_config
from gnn_rashomon.data.loaders import load_graph
from gnn_rashomon.fairness.metrics import fairness_metrics
from gnn_rashomon.rewiring.random_rewire import degree_sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recompute split-specific fairness columns from archived probabilities."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--input-table", required=True)
    parser.add_argument("--output-table", required=True)
    parser.add_argument("--graph-path", default=None)
    parser.add_argument("--sensitive-attribute", default="country")
    parser.add_argument("--positive-label", type=int, default=1)
    parser.add_argument("--degree-quantile", type=float, default=0.25)
    parser.add_argument(
        "--equalized-odds-combine",
        choices=["max", "sum", "average"],
        default="max",
    )
    return parser.parse_args()


def _to_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _predictions(path: str) -> np.ndarray:
    import torch

    probabilities = torch.load(path, map_location="cpu", weights_only=True)
    return probabilities.detach().cpu().numpy().argmax(axis=-1).astype(int)


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
    masks = {
        "validation": _to_numpy(data.val_mask).astype(bool) & label_mask & (labels >= 0),
        "test": _to_numpy(data.test_mask).astype(bool) & label_mask & (labels >= 0),
    }

    table = pd.read_csv(args.input_table)
    ambiguous = {
        "test_accuracy",
        "demographic_parity_gap",
        "equalized_odds_gap",
        "degree_disparity",
    }
    table = table.drop(columns=[column for column in ambiguous if column in table.columns])
    source = pd.read_csv(args.input_table)
    if "test_accuracy" in source:
        table["test_accuracy_report_only"] = source["test_accuracy"]

    for index, row in source.iterrows():
        predicted = _predictions(str(row["probability_path"]))
        for split, mask in masks.items():
            values = fairness_metrics(
                predicted,
                labels,
                sensitive,
                degree,
                mask,
                positive_label=args.positive_label,
                equalized_odds_combine=args.equalized_odds_combine,
                degree_quantile=args.degree_quantile,
            )
            for metric, value in values.items():
                column = (
                    f"validation_{metric}"
                    if split == "validation"
                    else f"test_{metric}_report_only"
                )
                table.loc[index, column] = value

    output_path = Path(args.output_table)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    print(f"Wrote split-specific fairness table to {output_path}")


if __name__ == "__main__":
    main()
