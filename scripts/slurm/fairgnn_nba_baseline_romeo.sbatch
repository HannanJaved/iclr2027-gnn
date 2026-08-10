#!/usr/bin/env bash
#SBATCH --job-name=gnn_fairgnn_nba
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
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

export MPLCONFIGDIR="${TMPDIR:-/tmp}/gnn-rashomon-matplotlib-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/gnn-rashomon-cache-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" .logs

NBA_SETS=(
  nba-gcn-seed-relative0.1-epochs200
  nba-gcn-hyperparameter-relative2-epochs200
)
METRICS=(demographic_parity_gap equalized_odds_gap degree_disparity)

echo "Building NBA fairness Pareto frontiers"
for set_id in "${NBA_SETS[@]}"; do
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

echo "Training FairGNN-style NBA baseline"
python -u -m gnn_rashomon.cli.run_fairgnn_baseline \
  --dataset nba \
  --seeds 42 43 44 45 46 \
  --epochs 2000 \
  --pretrain-epochs 200 \
  --hidden-channels 128 \
  --estimator-hidden-channels 128 \
  --dropout 0.5 \
  --learning-rate 0.001 \
  --weight-decay 1e-5 \
  --alpha 10 \
  --beta 1 \
  --sensitive-attribute country \
  --sensitive-train-count 50 \
  --min-validation-accuracy 0.70 \
  --min-validation-roc-auc 0.76 \
  --output-dir outputs

PARETO_MODEL_TABLES=()
for set_id in "${NBA_SETS[@]}"; do
  for metric in "${METRICS[@]}"; do
    PARETO_MODEL_TABLES+=("outputs/figures/${set_id}.fairness_${metric}.pareto.models.csv")
  done
done

echo "Comparing FairGNN baseline against NBA Rashomon Pareto frontiers"
python -u -m gnn_rashomon.cli.compare_fairgnn_baseline \
  --baseline-table outputs/baselines/fairgnn/nba_fairgnn_baseline_models.csv \
  --pareto-models "${PARETO_MODEL_TABLES[@]}" \
  --output-prefix outputs/figures/nba_fairgnn_pareto_baseline_comparison

python -u - <<'PY'
import json
from pathlib import Path

summary_path = Path("outputs/figures/nba_fairgnn_pareto_baseline_comparison.summary.json")
summary = json.loads(summary_path.read_text(encoding="utf-8"))
print("FairGNN comparison complete")
print(f"comparison_table={summary['comparison_table']}")
print(f"dominance_rate={summary['dominance_rate']:.3f}")
PY
