#!/bin/bash
#SBATCH --job-name=gnn_focal_padjust
#SBATCH --output=.logs/gnn_focal_padjust_%j.out
#SBATCH --error=.logs/gnn_focal_padjust_%j.err
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH -p romeo

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

python - <<'PY'
import csv
from pathlib import Path

rows = []

spearman_files = {
    "Cora": Path("outputs/metrics/cora-gcn-seed-relative0.1-epochs200.structure_spearman.csv"),
    "CiteSeer": Path("outputs/metrics/citeseer-gcn-seed-relative0.1-epochs200.structure_spearman.csv"),
    "PubMed": Path("outputs/metrics/pubmed-gcn-seed-relative0.1-epochs200.structure_spearman.csv"),
}

for dataset, path in spearman_files.items():
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["target"] == "rashomon_capacity" and row["feature"] == "local_homophily":
                rows.append(
                    {
                        "family": "homophily-capacity",
                        "dataset_model": dataset,
                        "test": "Spearman rho",
                        "direction": "negative",
                        "effect": row["spearman"],
                        "raw_p": float(row["p_value"]),
                    }
                )
            if row["target"] == "rashomon_capacity" and row["feature"] == "neighborhood_label_entropy":
                rows.append(
                    {
                        "family": "entropy-capacity",
                        "dataset_model": dataset,
                        "test": "Spearman rho",
                        "direction": "positive",
                        "effect": row["spearman"],
                        "raw_p": float(row["p_value"]),
                    }
                )

for dataset, p_value, effect in [
    ("Cora", 0.032, "+0.110"),
    ("CiteSeer", 0.032, "+0.104"),
    ("PubMed", 0.011, "+0.110"),
]:
    rows.append(
        {
            "family": "entropy-intervention",
            "dataset_model": dataset,
            "test": "Paired Wilcoxon",
            "direction": "positive",
            "effect": effect,
            "raw_p": p_value,
        }
    )

for dataset, p_value, effect in [
    ("Cora", 0.000999, "0.207"),
    ("CiteSeer", 0.007, "0.070"),
    ("PubMed", 0.049, "0.047"),
]:
    rows.append(
        {
            "family": "matched-regression",
            "dataset_model": dataset,
            "test": "Delta entropy -> Delta capacity",
            "direction": "positive",
            "effect": effect,
            "raw_p": p_value,
        }
    )

for dataset_model, p_value, effect, direction in [
    ("Cora GCN", 0.592, "0.354", "null"),
    ("Cora GAT", 0.0066, "0.874", "positive"),
    ("Cora APPNP", 0.122, "", "null"),
    ("PubMed GAT", 0.933, "", "null"),
    ("PubMed APPNP", 0.668, "", "null"),
]:
    rows.append(
        {
            "family": "explanation-regression",
            "dataset_model": dataset_model,
            "test": "Capacity coefficient",
            "direction": direction,
            "effect": effect,
            "raw_p": p_value,
        }
    )


def bh_adjust(p_values):
    n = len(p_values)
    order = sorted(range(n), key=lambda i: p_values[i])
    out = [None] * n
    running = 1.0
    for rank_from_end, idx in enumerate(reversed(order), start=1):
        rank = n - rank_from_end + 1
        running = min(running, p_values[idx] * n / rank)
        out[idx] = min(running, 1.0)
    return out


for family in sorted({row["family"] for row in rows}):
    indices = [i for i, row in enumerate(rows) if row["family"] == family]
    p_values = [rows[i]["raw_p"] for i in indices]
    q_values = bh_adjust(p_values)
    family_n = len(indices)
    for i, q_value in zip(indices, q_values):
        rows[i]["family_n"] = family_n
        rows[i]["bh_q"] = q_value
        rows[i]["bonferroni_p"] = min(rows[i]["raw_p"] * family_n, 1.0)
        rows[i]["bh_sig_0.05"] = q_value < 0.05
        rows[i]["bonf_sig_0.05"] = rows[i]["bonferroni_p"] < 0.05

out = Path("outputs/figures/stage1_focal_claim_correction.csv")
out.parent.mkdir(parents=True, exist_ok=True)
fieldnames = [
    "family",
    "family_n",
    "dataset_model",
    "test",
    "direction",
    "effect",
    "raw_p",
    "bh_q",
    "bonferroni_p",
    "bh_sig_0.05",
    "bonf_sig_0.05",
]
with out.open("w", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote focal claim correction table to {out}")
PY
