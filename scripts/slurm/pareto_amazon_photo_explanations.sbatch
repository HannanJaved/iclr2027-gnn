#!/usr/bin/env bash
#SBATCH --job-name=gnn_amz_xai_pareto
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
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

PREFIXES=(
  "amazon_photo-gcn-seed-relative0.1-epochs200"
  "amazon_photo-gat-arch_gat-relative0.1-epochs200"
  "amazon_photo-graphsage-arch_graphsage-relative0.1-epochs200"
)

for prefix in "${PREFIXES[@]}"; do
  metadata="outputs/explanations/${prefix}.gnnexplainer.metadata.json"
  test -f "${metadata}"
  python -u -m gnn_rashomon.cli.build_pareto \
    --mode explanation \
    --metadata "${metadata}" \
    --output-prefix "outputs/figures/${prefix}.gnnexplainer.pareto"
done

python -u - <<'PY'
import csv
import json
from pathlib import Path

prefixes = [
    "amazon_photo-gcn-seed-relative0.1-epochs200",
    "amazon_photo-gat-arch_gat-relative0.1-epochs200",
    "amazon_photo-graphsage-arch_graphsage-relative0.1-epochs200",
]

rows = []
for prefix in prefixes:
    metadata = json.loads(Path(f"outputs/explanations/{prefix}.gnnexplainer.metadata.json").read_text())
    regression_path = Path(metadata["outputs"]["regression"])
    pareto_summary_path = Path(f"outputs/figures/{prefix}.gnnexplainer.pareto.summary.json")
    pareto_summary = json.loads(pareto_summary_path.read_text())
    regression_rows = list(csv.DictReader(regression_path.open()))
    cap_rows = [
        row for row in regression_rows
        if row.get("outcome") == "top_k_jaccard_mean" and row.get("term") == "rashomon_capacity"
    ]
    cap = cap_rows[0] if cap_rows else {}
    rows.append(
        {
            "set_id": prefix,
            "selected_nodes": metadata["selected_node_count"],
            "selected_models": metadata["selected_model_count"],
            "pareto_models": pareto_summary["pareto_model_count"],
            "capacity_to_jaccard_coef": cap.get("coefficient", ""),
            "capacity_to_jaccard_p": cap.get("p_value", ""),
        }
    )

out = Path("outputs/figures/amazon_photo_explanation_pareto_summary.csv")
with out.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote Amazon-Photo explanation/Pareto summary to {out}")
PY
