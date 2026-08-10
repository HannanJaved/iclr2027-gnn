from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gnn_rashomon.multiplicity.predictive_entropy import predictive_entropy
from gnn_rashomon.multiplicity.rashomon_capacity import probability_diameter
from gnn_rashomon.multiplicity.variation_ratio import variation_ratio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic degree/homophily mechanism check for Rashomon multiplicity."
    )
    parser.add_argument("--nodes", type=int, default=400)
    parser.add_argument("--models", type=int, default=40)
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--homophily", nargs="+", type=float, default=[0.55, 0.65, 0.75, 0.85])
    parser.add_argument("--avg-degree", nargs="+", type=float, default=[4.0, 8.0, 16.0])
    parser.add_argument("--feature-signal", type=float, default=0.8)
    parser.add_argument("--message-signal", type=float, default=1.2)
    parser.add_argument("--model-noise", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--output-prefix", default="outputs/figures/synthetic_mechanism")
    return parser.parse_args()


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _binary_entropy(prob_one: np.ndarray) -> np.ndarray:
    probs = np.clip(prob_one, 1e-12, 1.0 - 1e-12)
    return -(probs * np.log(probs) + (1.0 - probs) * np.log(1.0 - probs))


def _generate_sbm(
    labels: np.ndarray,
    avg_degree: float,
    homophily: float,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(labels)
    same_prob = min(1.0, 2.0 * avg_degree * homophily / max(n - 1, 1))
    diff_prob = min(1.0, 2.0 * avg_degree * (1.0 - homophily) / max(n - 1, 1))
    same = labels[:, None] == labels[None, :]
    probs = np.where(same, same_prob, diff_prob)
    draws = rng.random((n, n)) < probs
    adjacency = np.triu(draws, k=1)
    adjacency = adjacency | adjacency.T
    return adjacency.astype(float)


def _simulate_probabilities(
    labels: np.ndarray,
    adjacency: np.ndarray,
    models: int,
    feature_signal: float,
    message_signal: float,
    model_noise: float,
    rng: np.random.Generator,
) -> np.ndarray:
    n = len(labels)
    centered_labels = labels * 2.0 - 1.0
    degree = adjacency.sum(axis=1)
    neighbor_sum = adjacency @ centered_labels
    neighbor_mean = np.divide(neighbor_sum, degree, out=np.zeros(n), where=degree > 0)
    probabilities = np.zeros((models, n, 2), dtype=float)
    for model_idx in range(models):
        feature_weight = rng.normal(feature_signal, 0.15)
        message_weight = rng.normal(message_signal, 0.25)
        node_noise = rng.normal(0.0, model_noise, size=n)
        logits = feature_weight * centered_labels + message_weight * neighbor_mean + node_noise
        prob_one = _sigmoid(logits)
        probabilities[model_idx, :, 1] = prob_one
        probabilities[model_idx, :, 0] = 1.0 - prob_one
    return probabilities


def _node_structure(labels: np.ndarray, adjacency: np.ndarray) -> pd.DataFrame:
    degree = adjacency.sum(axis=1)
    neighbor_positive = adjacency @ labels.astype(float)
    neighbor_rate = np.divide(
        neighbor_positive, degree, out=np.full(len(labels), np.nan), where=degree > 0
    )
    entropy = _binary_entropy(np.nan_to_num(neighbor_rate, nan=0.5))
    same_neighbors = adjacency @ labels.astype(float)
    same_rate = np.empty(len(labels), dtype=float)
    same_rate[labels == 1] = np.divide(
        same_neighbors[labels == 1],
        degree[labels == 1],
        out=np.zeros((labels == 1).sum()),
        where=degree[labels == 1] > 0,
    )
    same_rate[labels == 0] = np.divide(
        degree[labels == 0] - same_neighbors[labels == 0],
        degree[labels == 0],
        out=np.zeros((labels == 0).sum()),
        where=degree[labels == 0] > 0,
    )
    return pd.DataFrame(
        {
            "degree": degree,
            "log_degree": np.log1p(degree),
            "local_homophily": same_rate,
            "neighborhood_label_entropy": entropy,
        }
    )


def _spearman(frame: pd.DataFrame, left: str, right: str) -> float:
    clean = frame[[left, right]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(clean) < 3:
        return float("nan")
    return float(clean[left].rank().corr(clean[right].rank()))


def _run_cell(
    n: int,
    models: int,
    avg_degree: float,
    homophily: float,
    replicate: int,
    args: argparse.Namespace,
    rng: np.random.Generator,
) -> tuple[dict[str, object], pd.DataFrame]:
    labels = np.zeros(n, dtype=int)
    labels[n // 2 :] = 1
    rng.shuffle(labels)
    adjacency = _generate_sbm(labels, avg_degree=avg_degree, homophily=homophily, rng=rng)
    probabilities = _simulate_probabilities(
        labels=labels,
        adjacency=adjacency,
        models=models,
        feature_signal=args.feature_signal,
        message_signal=args.message_signal,
        model_noise=args.model_noise,
        rng=rng,
    )
    node_table = _node_structure(labels, adjacency)
    node_table["predictive_entropy"] = predictive_entropy(probabilities)
    node_table["variation_ratio"] = variation_ratio(probabilities)
    node_table["rashomon_capacity"] = probability_diameter(probabilities)
    predictions = probabilities.argmax(axis=-1)
    node_table["has_prediction_disagreement"] = (predictions != predictions[0:1]).any(axis=0)
    node_table["homophily_target"] = homophily
    node_table["avg_degree_target"] = avg_degree
    node_table["replicate"] = replicate

    summary = {
        "homophily_target": homophily,
        "avg_degree_target": avg_degree,
        "replicate": replicate,
        "realized_mean_degree": float(node_table["degree"].mean()),
        "realized_mean_local_homophily": float(node_table["local_homophily"].mean()),
        "mean_neighborhood_entropy": float(node_table["neighborhood_label_entropy"].mean()),
        "mean_rashomon_capacity": float(node_table["rashomon_capacity"].mean()),
        "disagreement_fraction": float(node_table["has_prediction_disagreement"].mean()),
        "degree_capacity_spearman": _spearman(node_table, "log_degree", "rashomon_capacity"),
        "homophily_capacity_spearman": _spearman(
            node_table, "local_homophily", "rashomon_capacity"
        ),
        "entropy_capacity_spearman": _spearman(
            node_table, "neighborhood_label_entropy", "rashomon_capacity"
        ),
    }
    return summary, node_table


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    summaries: list[dict[str, object]] = []
    nodes: list[pd.DataFrame] = []
    for avg_degree in args.avg_degree:
        for homophily in args.homophily:
            for replicate in range(args.replicates):
                summary, node_table = _run_cell(
                    n=args.nodes,
                    models=args.models,
                    avg_degree=avg_degree,
                    homophily=homophily,
                    replicate=replicate,
                    args=args,
                    rng=rng,
                )
                summaries.append(summary)
                nodes.append(node_table)

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    summary_table = pd.DataFrame(summaries)
    node_table = pd.concat(nodes, ignore_index=True)
    summary_path = Path(f"{prefix}.summary.csv")
    node_path = Path(f"{prefix}.nodes.csv")
    json_path = Path(f"{prefix}.summary.json")
    summary_table.to_csv(summary_path, index=False)
    node_table.to_csv(node_path, index=False)
    grouped = summary_table.groupby(["avg_degree_target", "homophily_target"], as_index=False).agg(
        mean_capacity=("mean_rashomon_capacity", "mean"),
        disagreement_fraction=("disagreement_fraction", "mean"),
        entropy_capacity_spearman=("entropy_capacity_spearman", "mean"),
        homophily_capacity_spearman=("homophily_capacity_spearman", "mean"),
        degree_capacity_spearman=("degree_capacity_spearman", "mean"),
    )
    grouped_path = Path(f"{prefix}.grid_summary.csv")
    grouped.to_csv(grouped_path, index=False)
    meta = {
        "summary_csv": str(summary_path),
        "node_csv": str(node_path),
        "grid_summary_csv": str(grouped_path),
        "nodes": args.nodes,
        "models": args.models,
        "replicates": args.replicates,
        "seed": args.seed,
        "note": (
            "Synthetic message-aggregate simulation with dialed homophily and average degree; "
            "no GNN training."
        ),
    }
    json_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote synthetic mechanism summaries to {summary_path}")
    print(f"Wrote synthetic mechanism grid summary to {grouped_path}")
    print(f"Wrote synthetic mechanism nodes to {node_path}")


if __name__ == "__main__":
    main()
