#!/usr/bin/env bash
#SBATCH --job-name=gnn_adult_fix_infer
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -p romeo
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR}
OUTPUT_ROOT="${OUTPUT_ROOT:?Export OUTPUT_ROOT from the corrected Adult pipeline}"
PERMUTATIONS="${PERMUTATIONS:-10000}"
SEED="${SEED:-20260715}"
GERMAN_CONCENTRATION_CSV="${GERMAN_CONCENTRATION_CSV:-${PROJECT_DIR}/outputs/figures/stage1_fairness_concentration_tests.csv}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES=""
export OUTPUT_ROOT PERMUTATIONS SEED GERMAN_CONCENTRATION_CSV

module purge
module load release/24.10 GCC/13.2.0 OpenMPI/4.1.6 PyTorch/2.3.0
# Optional: source your local cache/env setup here
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv/bin/activate"

python -u - <<'PY'
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

from gnn_rashomon.cli.config import load_config
from gnn_rashomon.data.loaders import load_graph

project = Path.cwd()
output_root = Path(os.environ["OUTPUT_ROOT"])
permutations = int(os.environ["PERMUTATIONS"])
seed = int(os.environ["SEED"])
config = load_config(["dataset=adult"], project / "configs")
graph = load_graph(
    "adult",
    root=str(project / config["dataset_config"]["root"]),
    dataset_config=config["dataset_config"],
)
sensitive = graph.sensitive_attributes["sex"].detach().cpu().numpy().astype(int)

sets = [
    ("Adult Seed", "adult-gcn-seed-relative0.1-epochs200"),
    ("Adult Hyperparameter", "adult-gcn-hyperparameter-relative2-epochs200"),
]
rows = []
for row_index, (claim, set_id) in enumerate(sets):
    table_path = output_root / "metrics" / f"{set_id}.multiplicity_nodes.csv"
    table = pd.read_csv(table_path)
    if len(table) != len(sensitive):
        raise SystemExit(
            f"{set_id}: multiplicity rows {len(table)} != sensitive rows {len(sensitive)}"
        )
    values = table["rashomon_capacity"].to_numpy(dtype=float)
    groups = sorted(np.unique(sensitive))
    if len(groups) != 2:
        raise SystemExit(f"Expected binary Adult sex attribute, got {groups}")
    group0 = values[sensitive == groups[0]]
    group1 = values[sensitive == groups[1]]
    difference = float(group1.mean() - group0.mean())
    test = mannwhitneyu(group1, group0, alternative="two-sided")

    rng = np.random.default_rng(seed + row_index)
    exceed = 0
    for _ in range(permutations):
        shuffled = rng.permutation(sensitive)
        permuted_difference = float(
            values[shuffled == groups[1]].mean() - values[shuffled == groups[0]].mean()
        )
        exceed += int(abs(permuted_difference) >= abs(difference))

    rows.append(
        {
            "claim": claim,
            "set_id": set_id,
            "group0": int(groups[0]),
            "group1": int(groups[1]),
            "group0_n": int(len(group0)),
            "group1_n": int(len(group1)),
            "group0_diameter_mean": float(group0.mean()),
            "group1_diameter_mean": float(group1.mean()),
            "mean_difference_group1_minus_group0": difference,
            "mannwhitney_u": float(test.statistic),
            "raw_p": float(test.pvalue),
            "permutation_p_two_sided": (exceed + 1) / (permutations + 1),
        }
    )

frame = pd.DataFrame(rows)
frame["bh_q"] = multipletests(frame["raw_p"], method="fdr_bh")[1]
frame["bonferroni_p"] = np.minimum(frame["raw_p"] * len(frame), 1.0)
frame["permutation_bh_q"] = multipletests(
    frame["permutation_p_two_sided"], method="fdr_bh"
)[1]
frame["permutation_bonferroni_p"] = np.minimum(
    frame["permutation_p_two_sided"] * len(frame), 1.0
)

figures = output_root / "figures"
figures.mkdir(parents=True, exist_ok=True)
csv_path = figures / "adult_corrected_fairness_concentration_tests.csv"
json_path = figures / "adult_corrected_fairness_concentration_tests_summary.json"
frame.to_csv(csv_path, index=False)
summary = {
    "passed": True,
    "rows": len(frame),
    "permutations": permutations,
    "seed": seed,
    "output_csv": str(csv_path),
    "correction_family": "two corrected Adult candidate sets",
}
json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(frame.to_string(index=False))
print(f"Wrote {csv_path}")
print(f"Wrote {json_path}")

german_path = Path(os.environ["GERMAN_CONCENTRATION_CSV"])
if not german_path.exists():
    raise SystemExit(f"Missing German concentration artifact: {german_path}")
german_source = pd.read_csv(german_path)
german = pd.DataFrame(
    {
        "claim": german_source["claim"],
        "set_id": german_source["set_id"],
        "group0": german_source["group0"],
        "group1": german_source["group1"],
        "group0_n": german_source["group0_n"],
        "group1_n": german_source["group1_n"],
        "group0_diameter_mean": german_source["group0_capacity_mean"],
        "group1_diameter_mean": german_source["group1_capacity_mean"],
        "mean_difference_group1_minus_group0": german_source[
            "mean_difference_group1_minus_group0"
        ],
        "mannwhitney_u": german_source["mannwhitney_u"],
        "raw_p": german_source["raw_p"],
        "permutation_p_two_sided": german_source["permutation_p_two_sided"],
    }
)
base_columns = list(german.columns)
combined = pd.concat([frame[base_columns], german], ignore_index=True)
combined["bh_q"] = multipletests(combined["raw_p"], method="fdr_bh")[1]
combined["bonferroni_p"] = np.minimum(combined["raw_p"] * len(combined), 1.0)
combined["permutation_bh_q"] = multipletests(
    combined["permutation_p_two_sided"], method="fdr_bh"
)[1]
combined["permutation_bonferroni_p"] = np.minimum(
    combined["permutation_p_two_sided"] * len(combined), 1.0
)

combined_csv = figures / "adult_german_fairness_concentration_tests.csv"
combined_json = figures / "adult_german_fairness_concentration_tests_summary.json"
combined.to_csv(combined_csv, index=False)
combined_summary = {
    "passed": True,
    "rows": len(combined),
    "adult_permutations": permutations,
    "adult_seed": seed,
    "german_source": str(german_path),
    "output_csv": str(combined_csv),
    "correction_family": "four corrected Adult and German Credit candidate sets",
}
combined_json.write_text(
    json.dumps(combined_summary, indent=2, sort_keys=True), encoding="utf-8"
)
print(combined.to_string(index=False))
print(f"Wrote {combined_csv}")
print(f"Wrote {combined_json}")
PY
