#!/usr/bin/env bash
#SBATCH --job-name=gnn_fair_conc
#SBATCH --time=00:30:00
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
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv-module/bin/activate"

PERMUTATIONS="${PERMUTATIONS:-10000}"
SEED="${SEED:-20260713}"
export PERMUTATIONS
export SEED

python - <<'PY'
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

from gnn_rashomon.cli.config import load_config
from gnn_rashomon.data.loaders import load_graph


ROOT = Path(".")
PROJECT_ROOT = Path(__file__).resolve().parents[3] if "__file__" in globals() else ROOT
OUT_CSV = ROOT / "outputs" / "figures" / "stage1_fairness_concentration_tests.csv"
OUT_JSON = ROOT / "outputs" / "figures" / "stage1_fairness_concentration_tests_summary.json"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

PERMUTATIONS = int(os.environ.get("PERMUTATIONS", "10000"))
SEED = int(os.environ.get("SEED", "20260713"))

SETS = [
    ("Adult Seed", "adult", "adult-gcn-seed-relative0.1-epochs200"),
    ("Adult Hyperparameter", "adult", "adult-gcn-hyperparameter-relative2-epochs200"),
    ("German Seed", "german_credit", "german_credit-gcn-seed-relative0.1-epochs200"),
    (
        "German Hyperparameter",
        "german_credit",
        "german_credit-gcn-hyperparameter-relative2-epochs200",
    ),
]


def to_numpy(value: object) -> np.ndarray:
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def load_sensitive(dataset: str, sensitive_attribute: str = "sex") -> np.ndarray:
    config = load_config([f"dataset={dataset}"], ROOT / "configs")
    graph = load_graph(
        dataset,
        root=str(ROOT / config["dataset_config"]["root"]),
        dataset_config=config["dataset_config"],
    )
    value = graph.sensitive_attributes.get(sensitive_attribute)
    if value is None and hasattr(graph.data, "sensitive_attr"):
        value = graph.data.sensitive_attr
    if value is None:
        raise SystemExit(f"No sensitive attribute {sensitive_attribute!r} for {dataset}.")
    return to_numpy(value).astype(int)


def permutation_p(values: np.ndarray, sensitive: np.ndarray, observed: float, seed: int) -> float:
    rng = np.random.default_rng(seed)
    exceed = 0
    labels = sensitive.copy()
    groups = sorted(np.unique(labels))
    if len(groups) != 2:
        return float("nan")
    observed_abs = abs(observed)
    for _ in range(PERMUTATIONS):
        shuffled = rng.permutation(labels)
        left = values[shuffled == groups[0]]
        right = values[shuffled == groups[1]]
        diff = float(right.mean() - left.mean())
        if abs(diff) >= observed_abs:
            exceed += 1
    return (exceed + 1) / (PERMUTATIONS + 1)


rows = []
sensitive_cache: dict[str, np.ndarray] = {}
for row_index, (label, dataset, set_id) in enumerate(SETS):
    path = ROOT / "outputs" / "metrics" / f"{set_id}.multiplicity_nodes.csv"
    if not path.exists():
        print(f"Skipping missing multiplicity table: {path}")
        continue
    if dataset not in sensitive_cache:
        sensitive_cache[dataset] = load_sensitive(dataset)
    sensitive = sensitive_cache[dataset]
    table = pd.read_csv(path)
    if len(table) != len(sensitive):
        raise SystemExit(f"{set_id}: multiplicity rows {len(table)} != sensitive rows {len(sensitive)}")

    values = table["rashomon_capacity"].to_numpy(dtype=float)
    groups = sorted(np.unique(sensitive))
    if len(groups) != 2:
        raise SystemExit(f"{set_id}: expected binary sensitive attribute, got {groups}")
    group0 = values[sensitive == groups[0]]
    group1 = values[sensitive == groups[1]]
    diff = float(group1.mean() - group0.mean())
    u = mannwhitneyu(group1, group0, alternative="two-sided")
    rows.append(
        {
            "claim": label,
            "dataset": dataset,
            "set_id": set_id,
            "sensitive_attribute": "sex",
            "group0": int(groups[0]),
            "group1": int(groups[1]),
            "group0_n": int(len(group0)),
            "group1_n": int(len(group1)),
            "group0_capacity_mean": float(group0.mean()),
            "group1_capacity_mean": float(group1.mean()),
            "mean_difference_group1_minus_group0": diff,
            "mannwhitney_u": float(u.statistic),
            "raw_p": float(u.pvalue),
            "permutation_p_two_sided": permutation_p(
                values,
                sensitive,
                diff,
                seed=SEED + row_index,
            ),
        }
    )

out = pd.DataFrame(rows)
if len(out):
    out["bh_q"] = multipletests(out["raw_p"].to_numpy(dtype=float), method="fdr_bh")[1]
    out["bonferroni_p"] = np.minimum(out["raw_p"].to_numpy(dtype=float) * len(out), 1.0)
    out["permutation_bh_q"] = multipletests(
        out["permutation_p_two_sided"].to_numpy(dtype=float),
        method="fdr_bh",
    )[1]
    out["permutation_bonferroni_p"] = np.minimum(
        out["permutation_p_two_sided"].to_numpy(dtype=float) * len(out),
        1.0,
    )
out.to_csv(OUT_CSV, index=False)

summary = {
    "output_csv": str(OUT_CSV),
    "rows": int(len(out)),
    "permutations": PERMUTATIONS,
    "seed": SEED,
    "note": (
        "Group1-minus-group0 is reported for the binary sex attribute used by the fairness "
        "experiments. Tests are node-level sensitivity checks; causal fairness claims remain "
        "out of scope."
    ),
}
OUT_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(f"Wrote fairness concentration tests to {OUT_CSV}")
print(f"Wrote summary to {OUT_JSON}")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
