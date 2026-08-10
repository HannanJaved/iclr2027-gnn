#!/usr/bin/env bash
#SBATCH --job-name=gnn_pg_report
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
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

export MPLCONFIGDIR="${TMPDIR:-/tmp}/gnn-rashomon-matplotlib-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/gnn-rashomon-cache-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" .logs

PREFIXES=(
  cora-gcn-seed-relative0.1-epochs200
  cora-gat-arch_gat-relative2-epochs200
  cora-appnp-arch_appnp-relative2-epochs200
  pubmed-gat-arch_gat-relative2-epochs200
  pubmed-appnp-arch_appnp-relative2-epochs200
)

for prefix in "${PREFIXES[@]}"; do
  metadata="outputs/explanations/${prefix}.pgexplainer.metadata.json"
  test -f "${metadata}"

  python -u -m gnn_rashomon.cli.build_explanation_report \
    --metadata "${metadata}" \
    --output-prefix "outputs/figures/${prefix}.pgexplainer"

  python -u -m gnn_rashomon.cli.build_pareto \
    --mode explanation \
    --metadata "${metadata}" \
    --output-prefix "outputs/figures/${prefix}.pgexplainer.pareto"
done

python -u - <<'PY'
import csv
import json
from pathlib import Path

prefixes = [
    "cora-gcn-seed-relative0.1-epochs200",
    "cora-gat-arch_gat-relative2-epochs200",
    "cora-appnp-arch_appnp-relative2-epochs200",
    "pubmed-gat-arch_gat-relative2-epochs200",
    "pubmed-appnp-arch_appnp-relative2-epochs200",
]

rows = []
for prefix in prefixes:
    metadata = json.loads(Path(f"outputs/explanations/{prefix}.pgexplainer.metadata.json").read_text())
    regression_path = Path(metadata["outputs"]["regression"])
    paper_summary_path = Path(f"outputs/figures/{prefix}.pgexplainer.paper_summary.csv")
    pareto_summary_path = Path(f"outputs/figures/{prefix}.pgexplainer.pareto.summary.json")
    paper_rows = list(csv.DictReader(paper_summary_path.open()))
    regression_rows = list(csv.DictReader(regression_path.open()))
    pareto_summary = json.loads(pareto_summary_path.read_text())
    cap_rows = [
        row for row in regression_rows
        if row.get("outcome") == "top_k_jaccard_mean" and row.get("term") == "rashomon_capacity"
    ]
    cap = cap_rows[0] if cap_rows else {}
    paper = paper_rows[0] if paper_rows else {}
    rows.append({
        "prefix": prefix,
        "dataset": metadata["dataset"],
        "selected_nodes": metadata["selected_node_count"],
        "selected_models": metadata["selected_model_count"],
        "mean_top_k_jaccard": paper.get("mean_top_k_jaccard", ""),
        "capacity_jaccard_spearman": paper.get("capacity_jaccard_spearman", ""),
        "capacity_jaccard_p_value": paper.get("capacity_jaccard_p_value", ""),
        "controlled_capacity_to_jaccard_coef": cap.get("coefficient", ""),
        "controlled_capacity_to_jaccard_p": cap.get("p_value", ""),
        "pareto_models": pareto_summary["pareto_model_count"],
    })

out = Path("outputs/figures/pgexplainer_explanation_pareto_summary.csv")
with out.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote PGExplainer explanation/Pareto summary to {out}")
PY
