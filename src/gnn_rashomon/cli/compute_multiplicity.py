from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from gnn_rashomon.cli.config import load_config
from gnn_rashomon.data.loaders import load_graph
from gnn_rashomon.multiplicity.predictive_entropy import predictive_entropy
from gnn_rashomon.multiplicity.rashomon_capacity import probability_diameter
from gnn_rashomon.multiplicity.summaries import metric_summary
from gnn_rashomon.multiplicity.variation_ratio import variation_ratio
from gnn_rashomon.rashomon.io import read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute node-level multiplicity for a Rashomon set."
    )
    parser.add_argument("rashomon_set", help="Path to Rashomon set JSON.")
    parser.add_argument("--graph-path", default=None)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument(
        "--capacity-method",
        choices=["exact", "chunked", "approximate"],
        default="exact",
    )
    parser.add_argument("--capacity-chunk-size", type=int, default=16)
    parser.add_argument("--capacity-sample-pairs", type=int, default=1000)
    parser.add_argument("--capacity-seed", type=int, default=0)
    return parser.parse_args()


def _load_probability(path: str) -> np.ndarray:
    import torch

    tensor = torch.load(path, map_location="cpu", weights_only=True)
    return tensor.detach().cpu().numpy()


def _to_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _split_names(data: object) -> np.ndarray:
    num_nodes = int(data.num_nodes)
    split = np.full(num_nodes, "unlabeled", dtype=object)
    for name, mask_name in [
        ("train", "train_mask"),
        ("validation", "val_mask"),
        ("test", "test_mask"),
    ]:
        mask = _to_numpy(getattr(data, mask_name)).astype(bool)
        split[mask] = name
    return split


def _degree(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    return np.bincount(edge_index[0].astype(int), minlength=num_nodes)


def _summaries_by_group(frame: pd.DataFrame, group_column: str) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for group, subset in frame.groupby(group_column, dropna=False):
        out[str(group)] = {
            "count": int(len(subset)),
            "fraction_prediction_disagreement": float(subset["has_prediction_disagreement"].mean()),
            "predictive_entropy": metric_summary(
                subset["predictive_entropy"].to_numpy(dtype=float)
            ),
            "variation_ratio": metric_summary(subset["variation_ratio"].to_numpy(dtype=float)),
            "rashomon_capacity": metric_summary(subset["rashomon_capacity"].to_numpy(dtype=float)),
        }
    return out


def _attach_graph_summaries(
    summary: dict[str, object],
    node_table: pd.DataFrame,
    dataset: str,
    graph_path: str | None,
) -> None:
    project_root = Path(__file__).resolve().parents[3]
    try:
        config = load_config([f"dataset={dataset}"], project_root / "configs")
        graph = load_graph(
            dataset,
            root=str(project_root / config["dataset_config"]["root"]),
            graph_path=graph_path,
            dataset_config=config["dataset_config"],
        )
    except Exception as exc:
        summary["conditional_summary_error"] = str(exc)
        return

    data = graph.data
    if int(data.num_nodes) != len(node_table):
        summary["conditional_summary_error"] = (
            f"Graph nodes ({int(data.num_nodes)}) do not match "
            f"multiplicity rows ({len(node_table)})."
        )
        return
    enriched = node_table.copy()
    enriched["label"] = _to_numpy(data.y).astype(int)
    enriched["split"] = _split_names(data)
    enriched["degree"] = _degree(_to_numpy(data.edge_index), int(data.num_nodes))
    enriched["degree_bin"] = pd.qcut(
        enriched["degree"].rank(method="first"),
        q=min(5, len(enriched)),
        labels=False,
        duplicates="drop",
    ).astype(int)
    summary["class_conditional"] = _summaries_by_group(enriched, "label")
    summary["split_specific"] = _summaries_by_group(enriched, "split")
    summary["degree_binned"] = _summaries_by_group(enriched, "degree_bin")


def main() -> None:
    args = parse_args()
    rashomon_path = Path(args.rashomon_set)
    rashomon = read_json(rashomon_path)
    retained = list(rashomon["retained_run_ids"])
    if len(retained) < 1:
        raise SystemExit(f"Rashomon set has no retained runs: {rashomon_path}")

    runs = rashomon["runs"]
    probabilities = np.stack(
        [_load_probability(runs[run_id]["probability_path"]) for run_id in retained]
    )
    entropy = predictive_entropy(probabilities)
    vr = variation_ratio(probabilities)
    capacity = probability_diameter(
        probabilities,
        method=args.capacity_method,
        chunk_size=args.capacity_chunk_size,
        sample_pairs=args.capacity_sample_pairs,
        seed=args.capacity_seed,
    )
    predictions = probabilities.argmax(axis=-1)
    disagreement = (predictions != predictions[0:1]).any(axis=0)

    node_table = pd.DataFrame(
        {
            "node_id": np.arange(probabilities.shape[1]),
            "predictive_entropy": entropy,
            "variation_ratio": vr,
            "rashomon_capacity": capacity,
            "has_prediction_disagreement": disagreement,
        }
    )
    set_id = rashomon["set_id"]
    output_dir = Path(args.output_dir)
    table_path = output_dir / "metrics" / f"{set_id}.multiplicity_nodes.csv"
    summary_path = output_dir / "metrics" / f"{set_id}.multiplicity_summary.json"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    node_table.to_csv(table_path, index=False)

    summary = {
        "set_id": set_id,
        "dataset": rashomon["dataset"],
        "architecture": rashomon["architecture"],
        "retained_count": len(retained),
        "num_nodes": int(probabilities.shape[1]),
        "num_classes": int(probabilities.shape[2]),
        "fraction_prediction_disagreement": float(disagreement.mean()),
        "predictive_entropy": metric_summary(entropy),
        "variation_ratio": metric_summary(vr),
        "rashomon_capacity": metric_summary(capacity),
        "rashomon_capacity_method": args.capacity_method,
    }
    _attach_graph_summaries(
        summary=summary,
        node_table=node_table,
        dataset=str(rashomon["dataset"]),
        graph_path=args.graph_path,
    )
    write_json(summary_path, summary)
    print(f"Wrote node multiplicity table to {table_path}")
    print(f"Wrote multiplicity summary to {summary_path}")
    print(
        f"retained={len(retained)} "
        f"disagreement_fraction={summary['fraction_prediction_disagreement']:.4f}"
    )


if __name__ == "__main__":
    main()
