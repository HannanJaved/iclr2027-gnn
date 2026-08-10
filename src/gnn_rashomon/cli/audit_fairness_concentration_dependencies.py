from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.stats.multitest import multipletests

from gnn_rashomon.cli.config import load_config
from gnn_rashomon.data.loaders import load_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit fairness-concentration claims with community-preserving permutations."
    )
    parser.add_argument("--adult-root", required=True)
    parser.add_argument(
        "--output-prefix",
        default="outputs/figures/fairness_concentration_dependency_audit",
    )
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def _sensitive_attribute(dataset: str, root: Path) -> np.ndarray:
    config = load_config([f"dataset={dataset}"], root / "configs")
    graph = load_graph(
        dataset,
        root=str(root / config["dataset_config"]["root"]),
        dataset_config=config["dataset_config"],
    )
    return graph.sensitive_attributes["sex"].detach().cpu().numpy().astype(int)


def _group_indices(groups: np.ndarray) -> list[np.ndarray]:
    return [np.flatnonzero(groups == group) for group in np.unique(groups)]


def _mean_difference(values: np.ndarray, sensitive: np.ndarray) -> float:
    groups = sorted(np.unique(sensitive))
    if len(groups) != 2:
        raise ValueError(f"Expected binary sensitive attribute, got {groups}")
    return float(values[sensitive == groups[1]].mean() - values[sensitive == groups[0]].mean())


def _community_permutation_p(
    values: np.ndarray,
    sensitive: np.ndarray,
    communities: np.ndarray,
    observed: float,
    permutations: int,
    seed: int,
) -> tuple[float, int, int]:
    indices = _group_indices(communities)
    permutable = [
        index for index in indices if len(index) >= 2 and np.unique(sensitive[index]).size >= 2
    ]
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(permutations):
        shuffled = sensitive.copy()
        for index in permutable:
            shuffled[index] = rng.permutation(shuffled[index])
        exceed += int(abs(_mean_difference(values, shuffled)) >= abs(observed))
    return (
        (exceed + 1) / (permutations + 1),
        int(sum(len(index) for index in permutable)),
        int(len(permutable)),
    )


def build_audit(root: Path, adult_root: Path, permutations: int, seed: int) -> pd.DataFrame:
    sets = [
        (
            "Adult Seed",
            "adult",
            "adult-gcn-seed-relative0.1-epochs200",
            adult_root / "metrics",
        ),
        (
            "Adult Hyperparameter",
            "adult",
            "adult-gcn-hyperparameter-relative2-epochs200",
            adult_root / "metrics",
        ),
        (
            "German Seed",
            "german_credit",
            "german_credit-gcn-seed-relative0.1-epochs200",
            root / "outputs" / "metrics",
        ),
        (
            "German Hyperparameter",
            "german_credit",
            "german_credit-gcn-hyperparameter-relative2-epochs200",
            root / "outputs" / "metrics",
        ),
    ]
    sensitive_cache: dict[str, np.ndarray] = {}
    rows: list[dict[str, object]] = []
    for position, (claim, dataset, set_id, metrics_root) in enumerate(sets):
        if dataset not in sensitive_cache:
            sensitive_cache[dataset] = _sensitive_attribute(dataset, root)
        sensitive = sensitive_cache[dataset]
        multiplicity = pd.read_csv(metrics_root / f"{set_id}.multiplicity_nodes.csv")
        structure = pd.read_csv(metrics_root / f"{set_id}.structure_nodes.csv")
        table = multiplicity[["node_id", "rashomon_capacity"]].merge(
            structure[["node_id", "community_id"]],
            on="node_id",
            how="inner",
            validate="one_to_one",
        )
        if len(table) != len(sensitive):
            raise ValueError(f"{set_id}: rows {len(table)} != sensitive values {len(sensitive)}")
        values = table["rashomon_capacity"].to_numpy(dtype=float)
        communities = table["community_id"].to_numpy()
        observed = _mean_difference(values, sensitive)
        p_value, permutable_nodes, permutable_communities = _community_permutation_p(
            values,
            sensitive,
            communities,
            observed,
            permutations,
            seed + position,
        )
        rows.append(
            {
                "claim": claim,
                "dataset": dataset,
                "set_id": set_id,
                "n": len(table),
                "communities": int(pd.Series(communities).nunique()),
                "permutable_nodes": permutable_nodes,
                "permutable_communities": permutable_communities,
                "mean_difference_group1_minus_group0": observed,
                "community_permutation_p_two_sided": p_value,
                "permutations": permutations,
            }
        )
    frame = pd.DataFrame(rows)
    p_values = frame["community_permutation_p_two_sided"].to_numpy(dtype=float)
    frame["community_permutation_bh_q"] = multipletests(p_values, method="fdr_bh")[1]
    frame["community_permutation_bonferroni_p"] = np.minimum(p_values * len(frame), 1.0)
    return frame


def main() -> None:
    args = parse_args()
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    frame = build_audit(Path("."), Path(args.adult_root), args.permutations, args.seed)
    csv_path = Path(f"{prefix}.csv")
    json_path = Path(f"{prefix}.json")
    frame.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(
            {
                "rows": len(frame),
                "permutations": args.permutations,
                "seed": args.seed,
                "output_csv": str(csv_path),
                "null": "sensitive labels shuffled within detected communities",
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    print(frame.to_string(index=False))
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
