#!/usr/bin/env bash
#SBATCH --job-name=gnn_struct_perm
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -p romeo
#SBATCH --requeue
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH=${PROJECT_DIR}/src:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES=""

module purge
module load release/24.10 GCC/13.2.0 OpenMPI/4.1.6 PyTorch/2.3.0

# Optional: source your local cache/env setup here
PROJECT_DIR=${PROJECT_DIR}
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv-module/bin/activate"

PERMUTATIONS="${PERMUTATIONS:-5000}"
SEED="${SEED:-20260713}"
export PERMUTATIONS
export SEED

python - <<'PY'
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


ROOT = Path(".")
OUT_CSV = ROOT / "outputs" / "figures" / "stage1_structural_permutation_audit.csv"
OUT_JSON = ROOT / "outputs" / "figures" / "stage1_structural_permutation_audit_summary.json"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

PERMUTATIONS = int(os.environ.get("PERMUTATIONS", "5000"))
SEED = int(os.environ.get("SEED", "20260713"))
MIN_GROUP_SIZE = 2

FOCAL = [
    (
        "Cora",
        ROOT / "outputs" / "metrics" / "cora-gcn-seed-relative0.1-epochs200.structure_nodes.csv",
    ),
    (
        "CiteSeer",
        ROOT / "outputs" / "metrics" / "citeseer-gcn-seed-relative0.1-epochs200.structure_nodes.csv",
    ),
    (
        "PubMed",
        ROOT / "outputs" / "metrics" / "pubmed-gcn-seed-relative0.1-epochs200.structure_nodes.csv",
    ),
]
PAIRS = [
    ("homophily-capacity", "rashomon_capacity", "local_homophily"),
    ("entropy-capacity", "rashomon_capacity", "neighborhood_label_entropy"),
]


def finite_pair(frame: pd.DataFrame, target: str, feature: str) -> pd.DataFrame:
    pair = frame[[target, feature, "degree", "community_id"]].copy()
    pair = pair.replace([np.inf, -np.inf], np.nan).dropna(subset=[target, feature])
    return pair.reset_index(drop=True)


def corr_from_ranks(x_rank: np.ndarray, y_rank: np.ndarray) -> float:
    if len(x_rank) < 3:
        return float("nan")
    x_centered = x_rank - float(np.mean(x_rank))
    y_centered = y_rank - float(np.mean(y_rank))
    denom = float(np.sqrt(np.sum(x_centered**2) * np.sum(y_centered**2)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(x_centered * y_centered) / denom)


def degree_bins(degree: pd.Series, max_bins: int = 10) -> np.ndarray:
    ranks = degree.rank(method="first")
    bins = min(max_bins, max(1, int(degree.nunique())))
    try:
        return pd.qcut(ranks, q=bins, labels=False, duplicates="drop").to_numpy()
    except ValueError:
        return np.zeros(len(degree), dtype=int)


def groups_for_mode(pair: pd.DataFrame, mode: str) -> np.ndarray:
    if mode == "global":
        return np.zeros(len(pair), dtype=object)
    if mode == "degree":
        return degree_bins(pair["degree"]).astype(object)
    if mode == "community":
        if "community_id" not in pair or pair["community_id"].isna().all():
            return np.zeros(len(pair), dtype=object)
        return pair["community_id"].astype(str).to_numpy()
    raise ValueError(f"Unknown mode: {mode}")


def group_indices(groups: np.ndarray) -> list[np.ndarray]:
    return [
        np.flatnonzero(groups == group)
        for group in np.unique(groups)
        if int(np.sum(groups == group)) >= MIN_GROUP_SIZE
    ]


def permute_within_groups(
    values: np.ndarray,
    groups: list[np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    permuted = values.copy()
    for idx in groups:
        permuted[idx] = rng.permutation(permuted[idx])
    return permuted


def permutation_p_value(
    pair: pd.DataFrame,
    target: str,
    feature: str,
    mode: str,
    seed: int,
    permutations: int,
) -> dict[str, float | int | str]:
    x = pair[target].to_numpy(dtype=float)
    y = pair[feature].to_numpy(dtype=float)
    x_rank = rankdata(x, method="average")
    y_rank = rankdata(y, method="average")
    observed = corr_from_ranks(x_rank, y_rank)
    groups = groups_for_mode(pair, mode)
    group_sizes = pd.Series(groups).value_counts()
    permutable_nodes = int(group_sizes[group_sizes >= MIN_GROUP_SIZE].sum())
    permutable_groups = int((group_sizes >= MIN_GROUP_SIZE).sum())
    indices = group_indices(groups)
    rng = np.random.default_rng(seed)

    if not math.isfinite(observed) or permutable_nodes < 3:
        return {
            "mode": mode,
            "observed_spearman": observed,
            "permutation_p_two_sided": float("nan"),
            "permutations": permutations,
            "null_abs_ge_observed": 0,
            "null_mean": float("nan"),
            "null_std": float("nan"),
            "permutable_nodes": permutable_nodes,
            "permutable_groups": permutable_groups,
            "groups": int(len(group_sizes)),
        }

    null = np.empty(permutations, dtype=float)
    exceed = 0
    observed_abs = abs(observed)
    for i in range(permutations):
        shuffled = permute_within_groups(x_rank, indices, rng)
        stat = corr_from_ranks(shuffled, y_rank)
        null[i] = stat
        if abs(stat) >= observed_abs:
            exceed += 1

    return {
        "mode": mode,
        "observed_spearman": observed,
        "permutation_p_two_sided": (exceed + 1) / (permutations + 1),
        "permutations": permutations,
        "null_abs_ge_observed": int(exceed),
        "null_mean": float(np.nanmean(null)),
        "null_std": float(np.nanstd(null, ddof=1)),
        "permutable_nodes": permutable_nodes,
        "permutable_groups": permutable_groups,
        "groups": int(len(group_sizes)),
    }


rows = []
for dataset, path in FOCAL:
    if not path.exists():
        print(f"Skipping missing node table: {path}")
        continue
    frame = pd.read_csv(path)
    if "community_id" not in frame.columns:
        frame["community_id"] = np.nan
    for family, target, feature in PAIRS:
        pair = finite_pair(frame, target, feature)
        x = pair[target].to_numpy(dtype=float)
        y = pair[feature].to_numpy(dtype=float)
        naive = spearmanr(x, y)
        for mode_index, mode in enumerate(["global", "degree", "community"]):
            result = permutation_p_value(
                pair,
                target,
                feature,
                mode=mode,
                seed=SEED + mode_index + 1000 * len(rows),
                permutations=PERMUTATIONS,
            )
            rows.append(
                {
                    "family": family,
                    "dataset": dataset,
                    "target": target,
                    "feature": feature,
                    "nobs": int(len(pair)),
                    "naive_spearman": float(naive.statistic),
                    "naive_p_value": float(naive.pvalue),
                    **result,
                }
            )

table = pd.DataFrame(rows)
table.to_csv(OUT_CSV, index=False)

summary = {
    "output_csv": str(OUT_CSV),
    "rows": int(len(table)),
    "permutations": PERMUTATIONS,
    "seed": SEED,
    "modes": ["global", "degree", "community"],
    "note": (
        "Permutation p-values shuffle Rashomon capacity within the requested null groups. "
        "The community mode preserves community membership when community_id is available; "
        "it is a practical graph-aware sensitivity check, not a full degree-preserving graph "
        "rewiring null."
    ),
}
OUT_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

print(f"Wrote structural permutation audit to {OUT_CSV}")
print(f"Wrote summary to {OUT_JSON}")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
