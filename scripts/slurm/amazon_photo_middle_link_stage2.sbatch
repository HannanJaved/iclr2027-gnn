#!/usr/bin/env bash
#SBATCH --job-name=gnn_amz_middle
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
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
import statsmodels.formula.api as smf
import torch
from statsmodels.stats.multitest import multipletests


ROOT = Path(".")
SETS = [
    ("Amazon-Photo GCN", "amazon_photo-gcn-seed-relative0.1-epochs200"),
    ("Amazon-Photo GAT", "amazon_photo-gat-arch_gat-relative0.1-epochs200"),
    ("Amazon-Photo GraphSAGE", "amazon_photo-graphsage-arch_graphsage-relative0.1-epochs200"),
]
OUT_REG = ROOT / "outputs" / "figures" / "amazon_photo_middle_link_regression.csv"
OUT_PERM = ROOT / "outputs" / "figures" / "amazon_photo_middle_link_permutation_audit.csv"
OUT_SUMMARY = ROOT / "outputs" / "figures" / "amazon_photo_middle_link_summary.json"
OUT_REG.parent.mkdir(parents=True, exist_ok=True)

PERMUTATIONS = int(os.environ.get("PERMUTATIONS", "5000"))
SEED = int(os.environ.get("SEED", "20260713"))
CONTROLS = [
    "log_degree",
    "local_homophily",
    "neighborhood_label_entropy",
    "feature_similarity",
    "confidence",
    "correct",
]


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


def regression(frame: pd.DataFrame, formula: str, label: str, outcome: str) -> pd.DataFrame:
    model = smf.ols(formula=formula, data=frame).fit(cov_type="HC3")
    table = model.summary2().tables[1].reset_index(names="term")
    table["label"] = label
    table["outcome"] = outcome
    table["formula"] = formula
    table["r_squared"] = float(model.rsquared)
    table["nobs"] = int(model.nobs)
    p_col = "P>|z|" if "P>|z|" in table.columns else "P>|t|"
    table["raw_p"] = table[p_col].astype(float)
    return table


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
    raise ValueError(mode)


def group_indices(groups: np.ndarray) -> list[np.ndarray]:
    return [
        np.flatnonzero(groups == group)
        for group in np.unique(groups)
        if int(np.sum(groups == group)) >= 2
    ]


def permute_within_groups(values: np.ndarray, indices: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    out = values.copy()
    for idx in indices:
        out[idx] = rng.permutation(out[idx])
    return out


def residualize(values: np.ndarray, q: np.ndarray) -> np.ndarray:
    return values - q @ (q.T @ values)


def coefficient(x: np.ndarray, residual_y: np.ndarray, q: np.ndarray) -> float:
    residual_x = residualize(x, q)
    denom = float(residual_x @ residual_x)
    if denom == 0.0:
        return float("nan")
    return float((residual_x @ residual_y) / denom)


def design_matrix(frame: pd.DataFrame) -> np.ndarray:
    controls = frame[CONTROLS].astype(float).to_numpy()
    design = np.column_stack([np.ones((len(frame), 1)), controls])
    q, _ = np.linalg.qr(design, mode="reduced")
    return q


def permutation_audit(frame: pd.DataFrame, mode: str, seed: int) -> dict[str, object]:
    clean = (
        frame[["rashomon_capacity", "probability_variance_mean", "degree", "community_id", *CONTROLS]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .reset_index(drop=True)
    )
    q = design_matrix(clean)
    y = clean["rashomon_capacity"].to_numpy(dtype=float)
    x = clean["probability_variance_mean"].to_numpy(dtype=float)
    residual_y = residualize(y, q)
    observed = coefficient(x, residual_y, q)
    groups = groups_for_mode(clean, mode)
    group_sizes = pd.Series(groups).value_counts()
    indices = group_indices(groups)
    rng = np.random.default_rng(seed)
    exceed = 0
    null = np.empty(PERMUTATIONS, dtype=float)
    for i in range(PERMUTATIONS):
        stat = coefficient(permute_within_groups(x, indices, rng), residual_y, q)
        null[i] = stat
        if math.isfinite(stat) and abs(stat) >= abs(observed):
            exceed += 1
    return {
        "mode": mode,
        "observed_probability_variance_coef": observed,
        "permutation_p_two_sided": (exceed + 1) / (PERMUTATIONS + 1),
        "permutations": PERMUTATIONS,
        "null_abs_ge_observed": int(exceed),
        "null_mean": float(np.nanmean(null)),
        "null_std": float(np.nanstd(null, ddof=1)),
        "permutable_nodes": int(group_sizes[group_sizes >= 2].sum()),
        "permutable_groups": int((group_sizes >= 2).sum()),
        "groups": int(len(group_sizes)),
    }


regression_rows = []
perm_rows = []
summaries = []
for set_index, (label, set_id) in enumerate(SETS):
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
            "predicted_class_probability_variance": prob_var_by_class[node_index, predicted],
            "retained_models": probs.shape[0],
        }
    )
    structure = pd.read_csv(structure_path)
    joined = structure.merge(middle, on="node_id", how="inner")
    joined["log_degree"] = np.log1p(joined["degree"].astype(float))
    if "community_id" not in joined.columns:
        joined["community_id"] = np.nan
    node_out = ROOT / "outputs" / "metrics" / f"{set_id}.middle_link_nodes.csv"
    joined.to_csv(node_out, index=False)

    regression_rows.append(
        regression(
            joined,
            "rashomon_capacity ~ probability_variance_mean + log_degree + local_homophily + "
            "neighborhood_label_entropy + feature_similarity + confidence + correct",
            label,
            "capacity_with_middle_link",
        )
    )
    for mode_index, mode in enumerate(["global", "degree", "community"]):
        result = permutation_audit(joined, mode=mode, seed=SEED + set_index * 100 + mode_index)
        perm_rows.append(
            {
                "label": label,
                "set_id": set_id,
                "outcome": "rashomon_capacity",
                "middle_metric": "probability_variance_mean",
                "controls": " + ".join(CONTROLS),
                "nobs": int(len(joined)),
                **result,
            }
        )
    summaries.append(
        {
            "label": label,
            "set_id": set_id,
            "retained_models": int(probs.shape[0]),
            "nodes": int(len(joined)),
            "middle_link_nodes_csv": str(node_out),
            "probability_variance_mean": float(joined["probability_variance_mean"].mean()),
        }
    )
    print(f"Wrote middle-link node table to {node_out}")

reg_out = pd.concat(regression_rows, ignore_index=True) if regression_rows else pd.DataFrame()
if len(reg_out):
    reg_out["bh_q_within_outcome"] = np.nan
    for outcome, subset in reg_out.groupby("outcome", sort=True):
        reg_out.loc[subset.index, "bh_q_within_outcome"] = multipletests(
            subset["raw_p"].to_numpy(dtype=float),
            method="fdr_bh",
        )[1]
reg_out.to_csv(OUT_REG, index=False)

perm_out = pd.DataFrame(perm_rows)
if len(perm_out):
    perm_out["bh_q_within_mode"] = np.nan
    perm_out["bonferroni_p_within_mode"] = np.nan
    for mode, subset in perm_out.groupby("mode", sort=True):
        p = subset["permutation_p_two_sided"].to_numpy(dtype=float)
        perm_out.loc[subset.index, "bh_q_within_mode"] = multipletests(p, method="fdr_bh")[1]
        perm_out.loc[subset.index, "bonferroni_p_within_mode"] = np.minimum(p * len(p), 1.0)
perm_out.to_csv(OUT_PERM, index=False)

summary = {
    "regression_csv": str(OUT_REG),
    "permutation_csv": str(OUT_PERM),
    "rows_regression": int(len(reg_out)),
    "rows_permutation": int(len(perm_out)),
    "permutations": PERMUTATIONS,
    "seed": SEED,
    "sets": summaries,
    "note": (
        "Amazon-Photo Stage-2 output-level middle-link extension. APPNP is excluded "
        "because the current APPNP Amazon-Photo set is diagnostic/degenerate."
    ),
}
OUT_SUMMARY.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
print(f"Wrote Amazon-Photo middle-link regression to {OUT_REG}")
print(f"Wrote Amazon-Photo middle-link permutation audit to {OUT_PERM}")
print(f"Wrote summary to {OUT_SUMMARY}")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
