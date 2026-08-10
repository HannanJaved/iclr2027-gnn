#!/usr/bin/env bash
#SBATCH --job-name=gnn_fair_pareto
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
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

SETS=(
  adult-gcn-seed-relative0.1-epochs200
  adult-gcn-hyperparameter-relative2-epochs200
  german_credit-gcn-seed-relative0.1-epochs200
  german_credit-gcn-hyperparameter-relative2-epochs200
)
METRICS=(demographic_parity_gap equalized_odds_gap degree_disparity)

for set_id in "${SETS[@]}"; do
  fairness_table="outputs/metrics/${set_id}.fairness_models.csv"
  test -f "${fairness_table}"

  for metric in "${METRICS[@]}"; do
    python -u -m gnn_rashomon.cli.build_pareto \
      --mode fairness \
      --fairness-table "${fairness_table}" \
      --fairness-metric "${metric}" \
      --output-prefix "outputs/figures/${set_id}.fairness_${metric}.pareto"
  done
done

python -u - <<'PY'
import csv
import json
from pathlib import Path

sets = [
    "adult-gcn-seed-relative0.1-epochs200",
    "adult-gcn-hyperparameter-relative2-epochs200",
    "german_credit-gcn-seed-relative0.1-epochs200",
    "german_credit-gcn-hyperparameter-relative2-epochs200",
]
metrics = ["demographic_parity_gap", "equalized_odds_gap", "degree_disparity"]
rows = []
for set_id in sets:
    for metric in metrics:
        summary_path = Path(f"outputs/figures/{set_id}.fairness_{metric}.pareto.summary.json")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "set_id": set_id,
                "fairness_metric": metric,
                "selected_model_count": summary["selected_model_count"],
                "pareto_model_count": summary["pareto_model_count"],
                "model_table": summary["model_table"],
                "baseline_table": summary["baseline_table"],
            }
        )

out = Path("outputs/figures/fairness_pareto_summary.csv")
with out.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print(f"Wrote fairness Pareto summary to {out}")
PY
