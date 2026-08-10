#!/usr/bin/env bash
#SBATCH --job-name=gnn_middle_perm
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
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
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
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
from statsmodels.stats.multitest import multipletests


ROOT = Path(".")
OUT_CSV = ROOT / "outputs" / "figures" / "stage1_middle_link_permutation_audit.csv"
OUT_JSON = ROOT / "outputs" / "figures" / "stage1_middle_link_permutation_audit_summary.json"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

PERMUTATIONS = int(os.environ.get("PERMUTATIONS", "5000"))
SEED = int(os.environ.get("SEED", "20260713"))
MIN_GROUP_SIZE = 2

SETS = [
    ("Cora", "cora-gcn-seed-relative0.1-epochs200"),
    ("CiteSeer", "citeseer-gcn-seed-relative0.1-epochs200"),
    ("PubMed", "pubmed-gcn-seed-relative0.1-epochs200"),
]

OUTCOME = "rashomon_capacity"
MIDDLE = "probability_variance_mean"
CONTROLS = [
    "log_degree",
    "local_homophily",
    "neighborhood_label_entropy",
    "feature_similarity",
    "confidence",
    "correct",
]


def degree_bins(degree: pd.Series, max_bins: int = 10) -> np.ndarray:
    ranks = degree.rank(method="first")
    bins = min(max_bins, max(1, int(degree.nunique())))
    try:
        return pd.qcut(ranks, q=bins, labels=False, duplicates="drop").to_numpy()
    except ValueError:
        return np.zeros(len(degree), dtype=int)


def groups_for_mode(frame: pd.DataFrame, mode: str) -> np.ndarray:
    if mode == "global":
        return np.zeros(len(frame), dtype=object)
    if mode == "degree":
        return degree_bins(frame["degree"]).astype(object)
    if mode == "community":
        if "community_id" not in frame or frame["community_id"].isna().all():
            return np.zeros(len(frame), dtype=object)
        return frame["community_id"].astype(str).to_numpy()
    raise ValueError(f"Unknown permutation mode: {mode}")


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


def residualize(values: np.ndarray, q: np.ndarray) -> np.ndarray:
    return values - q @ (q.T @ values)


def coefficient_from_residuals(x: np.ndarray, residual_y: np.ndarray, q: np.ndarray) -> float:
    residual_x = residualize(x, q)
    denom = float(residual_x @ residual_x)
    if denom == 0.0:
        return float("nan")
    return float((residual_x @ residual_y) / denom)


def prepare_frame(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "log_degree" not in frame.columns:
        frame["log_degree"] = np.log1p(frame["degree"].astype(float))
    if "community_id" not in frame.columns:
        frame["community_id"] = np.nan
    needed = [OUTCOME, MIDDLE, "degree", "community_id", *CONTROLS]
    return (
        frame[needed]
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=[OUTCOME, MIDDLE, *CONTROLS])
        .reset_index(drop=True)
    )


def design_matrix(frame: pd.DataFrame) -> np.ndarray:
    controls = frame[CONTROLS].astype(float).to_numpy()
    intercept = np.ones((len(frame), 1), dtype=float)
    design = np.column_stack([intercept, controls])
    q, _ = np.linalg.qr(design, mode="reduced")
    return q


def permutation_audit(
    frame: pd.DataFrame,
    mode: str,
    seed: int,
    permutations: int,
) -> dict[str, float | int | str]:
    y = frame[OUTCOME].astype(float).to_numpy()
    x = frame[MIDDLE].astype(float).to_numpy()
    q = design_matrix(frame)
    residual_y = residualize(y, q)
    observed = coefficient_from_residuals(x, residual_y, q)

    groups = groups_for_mode(frame, mode)
    group_sizes = pd.Series(groups).value_counts()
    indices = group_indices(groups)
    permutable_nodes = int(group_sizes[group_sizes >= MIN_GROUP_SIZE].sum())
    permutable_groups = int((group_sizes >= MIN_GROUP_SIZE).sum())

    rng = np.random.default_rng(seed)
    null = np.empty(permutations, dtype=float)
    exceed = 0
    observed_abs = abs(observed)
    for i in range(permutations):
        shuffled_x = permute_within_groups(x, indices, rng)
        stat = coefficient_from_residuals(shuffled_x, residual_y, q)
        null[i] = stat
        if math.isfinite(stat) and abs(stat) >= observed_abs:
            exceed += 1

    return {
        "mode": mode,
        "observed_probability_variance_coef": observed,
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
for dataset_index, (dataset, set_id) in enumerate(SETS):
    path = ROOT / "outputs" / "metrics" / f"{set_id}.middle_link_nodes.csv"
    if not path.exists():
        print(f"Skipping missing middle-link node table: {path}")
        continue
    frame = prepare_frame(path)
    for mode_index, mode in enumerate(["global", "degree", "community"]):
        result = permutation_audit(
            frame,
            mode=mode,
            seed=SEED + dataset_index * 100 + mode_index,
            permutations=PERMUTATIONS,
        )
        rows.append(
            {
                "family": "middle-link-capacity",
                "dataset": dataset,
                "set_id": set_id,
                "outcome": OUTCOME,
                "middle_metric": MIDDLE,
                "controls": " + ".join(CONTROLS),
                "nobs": int(len(frame)),
                **result,
            }
        )

out = pd.DataFrame(rows)
if len(out):
    out["bh_q_within_mode"] = np.nan
    out["bonferroni_p_within_mode"] = np.nan
    for mode, subset in out.groupby("mode", sort=True):
        p = subset["permutation_p_two_sided"].to_numpy(dtype=float)
        out.loc[subset.index, "bh_q_within_mode"] = multipletests(p, method="fdr_bh")[1]
        out.loc[subset.index, "bonferroni_p_within_mode"] = np.minimum(p * len(p), 1.0)
out.to_csv(OUT_CSV, index=False)

summary = {
    "output_csv": str(OUT_CSV),
    "rows": int(len(out)),
    "permutations": PERMUTATIONS,
    "seed": SEED,
    "modes": ["global", "degree", "community"],
    "note": (
        "Permutation p-values shuffle the middle-link probability-variance metric within "
        "the requested null groups, then recompute the residualized OLS coefficient for "
        "probability variance in the controlled Rashomon-capacity regression. This audits "
        "the output-level variance bridge, not hidden message-aggregate variance."
    ),
}
OUT_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

print(f"Wrote middle-link permutation audit to {OUT_CSV}")
print(f"Wrote summary to {OUT_JSON}")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
