from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gnn_rashomon.analysis.structural import (
    regression_table,
    residual_diagnostics,
    spearman_table,
)
from gnn_rashomon.cli.config import load_config
from gnn_rashomon.data.loaders import load_graph
from gnn_rashomon.rashomon.io import read_json
from gnn_rashomon.structure.community_boundaries import boundary_features, clustering_coefficient
from gnn_rashomon.structure.degree import node_degree
from gnn_rashomon.structure.feature_alignment import node_feature_similarity
from gnn_rashomon.structure.homophily import local_homophily
from gnn_rashomon.structure.neighborhood_entropy import neighborhood_label_entropy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Join structural graph metrics with multiplicity outputs.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--rashomon-set", required=True)
    parser.add_argument("--multiplicity-table", required=True)
    parser.add_argument("--graph-path", default=None)
    parser.add_argument("--include-community", action="store_true")
    parser.add_argument(
        "--profile",
        choices=["full", "scalable"],
        default="full",
        help="Scalable omits clustering, feature similarity, and community detection.",
    )
    parser.add_argument("--community-seed", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs")
    return parser.parse_args()


def _to_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _split_names(data: object) -> np.ndarray:
    num_nodes = int(data.num_nodes)
    split = np.full(num_nodes, "unlabeled", dtype=object)
    for name, mask_name in [("train", "train_mask"), ("validation", "val_mask"), ("test", "test_mask")]:
        mask = _to_numpy(getattr(data, mask_name)).astype(bool)
        split[mask] = name
    return split


def _write_scatter_plots(joined: pd.DataFrame, set_id: str, figures_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures_dir.mkdir(parents=True, exist_ok=True)
    plots = [
        (
            "degree",
            "rashomon_capacity",
            "Node degree",
            "Rashomon capacity",
            f"{set_id}.degree_vs_capacity.png",
        ),
        (
            "neighborhood_label_entropy",
            "rashomon_capacity",
            "Neighborhood label entropy",
            "Rashomon capacity",
            f"{set_id}.entropy_vs_capacity.png",
        ),
    ]
    for x_col, y_col, x_label, y_label, filename in plots:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(
            joined[x_col].to_numpy(dtype=float),
            joined[y_col].to_numpy(dtype=float),
            s=8,
            alpha=0.35,
            linewidths=0,
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(figures_dir / filename, dpi=180)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]
    output_dir = Path(args.output_dir)
    config = load_config([f"dataset={args.dataset}"], project_root / "configs")
    graph = load_graph(
        args.dataset,
        root=str(project_root / config["dataset_config"]["root"]),
        graph_path=args.graph_path,
        dataset_config=config["dataset_config"],
    )
    data = graph.data

    edge_index = _to_numpy(data.edge_index).astype(int)
    labels = _to_numpy(data.y).astype(int)
    if hasattr(data, "label_mask"):
        label_mask = _to_numpy(getattr(data, "label_mask")).astype(bool) & (labels >= 0)
    else:
        label_mask = labels >= 0
    num_nodes = int(data.num_nodes)
    num_classes = int(graph.metadata.num_classes)
    degree = node_degree(edge_index, num_nodes)
    scalable = args.profile == "scalable"
    if scalable and args.include_community:
        raise SystemExit("--include-community is not supported with --profile scalable.")
    clustering = (
        np.full(num_nodes, np.nan, dtype=float)
        if scalable
        else clustering_coefficient(edge_index, num_nodes)
    )
    homophily = local_homophily(edge_index, labels, num_nodes, label_mask=label_mask)
    entropy = neighborhood_label_entropy(
        edge_index,
        labels,
        num_nodes,
        num_classes,
        label_mask=label_mask,
    )
    feature_similarity = (
        np.full(num_nodes, np.nan, dtype=float)
        if scalable
        else node_feature_similarity(edge_index, _to_numpy(data.x).astype(float), num_nodes)
    )

    multiplicity = pd.read_csv(args.multiplicity_table)
    if len(multiplicity) != num_nodes:
        raise SystemExit(
            f"Multiplicity rows ({len(multiplicity)}) do not match graph nodes ({num_nodes})."
        )

    rashomon = read_json(Path(args.rashomon_set))
    retained = list(rashomon["retained_run_ids"])
    runs = rashomon["runs"]
    import torch

    probabilities = [
        torch.load(runs[run_id]["probability_path"], map_location="cpu", weights_only=True)
        .detach()
        .cpu()
        .numpy()
        for run_id in retained
    ]
    mean_probabilities = np.mean(np.stack(probabilities), axis=0)
    predicted_class = mean_probabilities.argmax(axis=1)
    confidence = mean_probabilities.max(axis=1)
    correct = np.where(label_mask, predicted_class == labels, np.nan)

    joined = multiplicity.copy()
    joined["degree"] = degree
    joined["clustering_coefficient"] = clustering
    joined["local_homophily"] = homophily
    joined["local_heterophily"] = 1.0 - homophily
    joined["neighborhood_label_entropy"] = entropy
    joined["feature_similarity"] = feature_similarity
    joined["label"] = labels
    joined["has_label"] = label_mask.astype(int)
    joined["split"] = _split_names(data)
    joined["predicted_class"] = predicted_class
    joined["confidence"] = confidence
    joined["correct"] = correct
    joined["is_isolated"] = (degree == 0).astype(int)
    if args.include_community:
        community = boundary_features(edge_index, num_nodes, seed=args.community_seed)
        for name, values in community.items():
            joined[name] = values

    set_id = rashomon["set_id"]
    node_path = output_dir / "metrics" / f"{set_id}.structure_nodes.csv"
    corr_path = output_dir / "metrics" / f"{set_id}.structure_spearman.csv"
    reg_path = output_dir / "metrics" / f"{set_id}.structure_regression.csv"
    residual_path = output_dir / "metrics" / f"{set_id}.structure_residual_diagnostics.csv"
    summary_path = output_dir / "metrics" / f"{set_id}.structure_summary.json"
    node_path.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(node_path, index=False)

    targets = ["predictive_entropy", "variation_ratio", "rashomon_capacity"]
    structural_features = [
        "degree",
        "local_homophily",
        "local_heterophily",
        "neighborhood_label_entropy",
        "confidence",
        "correct",
    ]
    if not scalable:
        structural_features[1:1] = ["clustering_coefficient"]
        structural_features.insert(-2, "feature_similarity")
    if args.include_community:
        structural_features.extend(
            [
                "is_community_boundary",
                "cross_community_fraction",
                "distance_to_community_boundary",
            ]
        )
    correlations = spearman_table(joined, targets, structural_features)
    correlations.to_csv(corr_path, index=False)
    regression_predictors = [
        "log_degree",
        "local_homophily",
        "neighborhood_label_entropy",
        "confidence",
        "correct",
    ]
    if not scalable:
        regression_predictors.insert(3, "feature_similarity")
    regressions = pd.concat(
        [regression_table(joined, target, regression_predictors) for target in targets]
    )
    regressions.to_csv(reg_path, index=False)
    residuals = pd.DataFrame(
        [residual_diagnostics(joined, target, regression_predictors) for target in targets]
    )
    residuals.to_csv(residual_path, index=False)
    _write_scatter_plots(joined, set_id, output_dir / "figures")

    summary = {
        "set_id": set_id,
        "dataset": args.dataset,
        "num_nodes": num_nodes,
        "retained_count": len(retained),
        "labeled_count": int(label_mask.sum()),
        "unlabeled_count": int((~label_mask).sum()),
        "mean_degree": float(np.mean(degree)),
        "structural_profile": args.profile,
        "mean_clustering_coefficient": None if scalable else float(np.mean(clustering)),
        "mean_local_homophily": float(np.nanmean(homophily)),
        "mean_neighborhood_label_entropy": float(np.nanmean(entropy)),
        "mean_feature_similarity": None if scalable else float(np.mean(feature_similarity)),
        "mean_confidence": float(np.mean(confidence)),
        "mean_correct": float(np.nanmean(correct)),
        "include_community": bool(args.include_community),
    }
    if args.include_community:
        summary["mean_is_community_boundary"] = float(joined["is_community_boundary"].mean())
        summary["mean_cross_community_fraction"] = float(joined["cross_community_fraction"].mean())
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote joined node table to {node_path}")
    print(f"Wrote Spearman table to {corr_path}")
    print(f"Wrote regression table to {reg_path}")
    print(f"Wrote residual diagnostics to {residual_path}")
    print(f"Wrote summary to {summary_path}")


if __name__ == "__main__":
    main()
