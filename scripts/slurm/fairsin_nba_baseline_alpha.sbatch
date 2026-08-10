#!/usr/bin/env bash
#SBATCH --job-name=gnn_fairsin_nba
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH -p alpha
#SBATCH --requeue
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH=${PROJECT_DIR}/src:${PYTHONPATH:-}
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
module purge
module load release/24.10 GCC/13.2.0 OpenMPI/4.1.6 PyTorch/2.3.0
# Optional: source your local cache/env setup here
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv/bin/activate"

python -u -m gnn_rashomon.cli.run_fairsin_baseline \
  --dataset nba --seeds 42 43 44 45 46 \
  --deltas 0.1 0.5 1.0 2.0 \
  --epochs 500 --f3-pretrain-epochs 200 \
  --hidden-channels 128 --dropout 0.5 \
  --learning-rate 0.001 --weight-decay 1e-5 \
  --f3-weight 1 --adversary-weight 1 \
  --sensitive-attribute country --device cuda

PARETO_MODELS=()
for set_id in nba-gcn-seed-relative0.1-epochs200 nba-gcn-hyperparameter-relative2-epochs200; do
  for metric in demographic_parity_gap equalized_odds_gap degree_disparity; do
    PARETO_MODELS+=("outputs/figures/${set_id}.fairness_${metric}.pareto.models.csv")
  done
done
python -u -m gnn_rashomon.cli.compare_fairgnn_baseline \
  --baseline-table outputs/baselines/fairsin/nba_fairsin_baseline_models.csv \
  --pareto-models "${PARETO_MODELS[@]}" \
  --output-prefix outputs/figures/nba_fairsin_pareto_baseline_comparison
