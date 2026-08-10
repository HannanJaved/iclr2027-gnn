#!/usr/bin/env bash
#SBATCH --job-name=gnn_arch_analysis
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
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

export MPLCONFIGDIR="${TMPDIR:-/tmp}/gnn-rashomon-matplotlib-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/gnn-rashomon-cache-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

DATASETS=(cora pubmed)
ARCHITECTURES=(gat graphsage appnp)
EPSILON=0.10
SET_SUFFIX=relative0.1-epochs200

for dataset in "${DATASETS[@]}"; do
  for architecture in "${ARCHITECTURES[@]}"; do
    set_path="outputs/rashomon_sets/${dataset}-${architecture}-arch_${architecture}-${SET_SUFFIX}.json"
    mult_table="outputs/metrics/${dataset}-${architecture}-arch_${architecture}-${SET_SUFFIX}.multiplicity_nodes.csv"

    python -u -m gnn_rashomon.cli.generate_rashomon \
      --dataset "${dataset}" \
      --architecture "${architecture}" \
      --set-type "arch_${architecture}" \
      --rashomon-type "arch_${architecture}" \
      --max-epochs 200 \
      --tolerance-mode relative \
      --epsilon "${EPSILON}"

    python -u -m gnn_rashomon.cli.compute_multiplicity "${set_path}"

    python -u -m gnn_rashomon.cli.compute_structure \
      --dataset "${dataset}" \
      --rashomon-set "${set_path}" \
      --multiplicity-table "${mult_table}" \
      --include-community
  done
done

python -u - <<'PY'
import csv
import json
from pathlib import Path

root = Path("outputs")
rows = []
for dataset in ["cora", "pubmed"]:
    for architecture in ["gat", "graphsage", "appnp"]:
        prefix = f"{dataset}-{architecture}-arch_{architecture}-relative0.1-epochs200"
        set_path = root / "rashomon_sets" / f"{prefix}.json"
        summary_path = root / "metrics" / f"{prefix}.multiplicity_summary.json"
        structure_path = root / "metrics" / f"{prefix}.structure_summary.json"
        if not set_path.exists() or not summary_path.exists():
            continue
        rashomon = json.loads(set_path.read_text(encoding="utf-8"))
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        structure = json.loads(structure_path.read_text(encoding="utf-8")) if structure_path.exists() else {}
        rows.append(
            {
                "dataset": dataset,
                "architecture": architecture,
                "candidate_count": rashomon["candidate_count"],
                "retained_count": rashomon["retained_count"],
                "baseline_loss": rashomon["baseline_loss"],
                "threshold": rashomon["threshold"],
                "disagreement_fraction": summary["fraction_prediction_disagreement"],
                "capacity_mean": summary["rashomon_capacity"]["mean"],
                "entropy_mean": summary["predictive_entropy"]["mean"],
                "variation_ratio_mean": summary["variation_ratio"]["mean"],
                "mean_confidence": structure.get("mean_confidence", ""),
                "mean_correct": structure.get("mean_correct", ""),
            }
        )

out = root / "figures" / "architecture_validation_seed_relative0.1_summary.csv"
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["dataset", "architecture"])
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote architecture validation summary to {out}")
PY
