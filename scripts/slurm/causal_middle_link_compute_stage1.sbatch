#!/usr/bin/env bash
#SBATCH --job-name=gnn_middle_compute
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
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

python - <<'PY'
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import torch
from statsmodels.stats.multitest import multipletests


ROOT = Path(".")
SETS = [
    ("Cora", "cora-gcn-seed-relative0.1-epochs200"),
    ("CiteSeer", "citeseer-gcn-seed-relative0.1-epochs200"),
    ("PubMed", "pubmed-gcn-seed-relative0.1-epochs200"),
]
OUT_REG = ROOT / "outputs" / "figures" / "stage1_causal_middle_link_regression.csv"
OUT_SUMMARY = ROOT / "outputs" / "figures" / "stage1_causal_middle_link_regression_summary.json"
OUT_REG.parent.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_probabilities(rashomon: dict) -> np.ndarray:
    arrays = []
    for run_id in rashomon["retained_run_ids"]:
        path = rashomon["runs"][run_id]["probability_path"]
        arr = torch.load(path, map_location="cpu", weights_only=True).detach().cpu().numpy()
        arrays.append(arr.astype(np.float64))
    if not arrays:
        raise ValueError(f"No retained probability arrays for {rashomon.get('set_id')}")
    return np.stack(arrays, axis=0)


def regression(frame: pd.DataFrame, formula: str, dataset: str, outcome: str) -> pd.DataFrame:
    model = smf.ols(formula=formula, data=frame).fit(cov_type="HC3")
    table = model.summary2().tables[1].reset_index(names="term")
    table["dataset"] = dataset
    table["outcome"] = outcome
    table["formula"] = formula
    table["r_squared"] = float(model.rsquared)
    table["nobs"] = int(model.nobs)
    p_col = "P>|z|" if "P>|z|" in table.columns else "P>|t|"
    table["raw_p"] = table[p_col].astype(float)
    return table


rows = []
summaries = []
for dataset, set_id in SETS:
    rashomon_path = ROOT / "outputs" / "rashomon_sets" / f"{set_id}.json"
    structure_path = ROOT / "outputs" / "metrics" / f"{set_id}.structure_nodes.csv"
    if not rashomon_path.exists() or not structure_path.exists():
        print(f"Skipping missing inputs for {set_id}")
        continue

    rashomon = load_json(rashomon_path)
    probs = load_probabilities(rashomon)
    mean_probs = probs.mean(axis=0)
    prob_var_by_class = probs.var(axis=0, ddof=1) if probs.shape[0] > 1 else np.zeros_like(mean_probs)
    predicted = mean_probs.argmax(axis=1)
    node_index = np.arange(mean_probs.shape[0])

    middle = pd.DataFrame(
        {
            "node_id": node_index,
            "probability_variance_mean": prob_var_by_class.mean(axis=1),
            "probability_variance_max": prob_var_by_class.max(axis=1),
            "predicted_class_probability_variance": prob_var_by_class[node_index, predicted],
            "mean_confidence": mean_probs.max(axis=1),
            "retained_models": probs.shape[0],
        }
    )
    structure = pd.read_csv(structure_path)
    joined = structure.merge(middle, on="node_id", how="inner")
    joined["log_degree"] = np.log1p(joined["degree"].astype(float))
    node_out = ROOT / "outputs" / "metrics" / f"{set_id}.middle_link_nodes.csv"
    joined.to_csv(node_out, index=False)

    formulas = [
        (
            "probability_variance_mean",
            "probability_variance_mean ~ log_degree + local_homophily + "
            "neighborhood_label_entropy + feature_similarity + confidence + correct",
        ),
        (
            "predicted_class_probability_variance",
            "predicted_class_probability_variance ~ log_degree + local_homophily + "
            "neighborhood_label_entropy + feature_similarity + confidence + correct",
        ),
        (
            "capacity_with_middle_link",
            "rashomon_capacity ~ probability_variance_mean + log_degree + local_homophily + "
            "neighborhood_label_entropy + feature_similarity + confidence + correct",
        ),
    ]
    for outcome, formula in formulas:
        rows.append(regression(joined, formula, dataset, outcome))

    summaries.append(
        {
            "dataset": dataset,
            "set_id": set_id,
            "retained_models": int(probs.shape[0]),
            "nodes": int(len(joined)),
            "middle_link_nodes_csv": str(node_out),
            "probability_variance_mean": float(joined["probability_variance_mean"].mean()),
            "predicted_class_probability_variance_mean": float(
                joined["predicted_class_probability_variance"].mean()
            ),
        }
    )
    print(f"Wrote middle-link node table to {node_out}")

out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
if len(out):
    out["bh_q_within_outcome"] = np.nan
    for outcome, subset in out.groupby("outcome", sort=True):
        out.loc[subset.index, "bh_q_within_outcome"] = multipletests(
            subset["raw_p"].to_numpy(dtype=float),
            method="fdr_bh",
        )[1]
out.to_csv(OUT_REG, index=False)

summary = {
    "output_csv": str(OUT_REG),
    "datasets": summaries,
    "rows": int(len(out)),
    "note": (
        "Middle-link proxy is per-node probability variance across retained Rashomon models. "
        "This measures predictive distribution variability, not hidden embedding/message "
        "variance."
    ),
}
OUT_SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(f"Wrote middle-link regression table to {OUT_REG}")
print(f"Wrote summary to {OUT_SUMMARY}")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
