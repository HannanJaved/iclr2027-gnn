#!/usr/bin/env bash
#SBATCH --job-name=gnn_budget_fair
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -p romeo
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES=""

module purge
module load release/24.10 GCC/13.2.0 OpenMPI/4.1.6 PyTorch/2.3.0
# Optional: source your local cache/env setup here
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv-module/bin/activate"

CANDIDATES=(
  outputs/figures/nba-gcn-seed-relative0.1-epochs200.fairness_demographic_parity_gap.pareto.models.csv
  outputs/figures/nba-gcn-seed-relative0.1-epochs200.fairness_equalized_odds_gap.pareto.models.csv
  outputs/figures/nba-gcn-seed-relative0.1-epochs200.fairness_degree_disparity.pareto.models.csv
  outputs/figures/nba-gcn-hyperparameter-relative2-epochs200.fairness_demographic_parity_gap.pareto.models.csv
  outputs/figures/nba-gcn-hyperparameter-relative2-epochs200.fairness_equalized_odds_gap.pareto.models.csv
  outputs/figures/nba-gcn-hyperparameter-relative2-epochs200.fairness_degree_disparity.pareto.models.csv
)

python -u -m gnn_rashomon.cli.compare_budget_matched_fairness \
  --baseline-table outputs/baselines/fairgnn/nba_fairgnn_baseline_models.csv \
  --candidate-models "${CANDIDATES[@]}" \
  --sample-size 5 \
  --draws 1000 \
  --seed 20260716 \
  --output-prefix outputs/budget_matched_fairness/nba_gcn_vs_fairgnn_k5

python -u -m gnn_rashomon.cli.compare_budget_matched_fairness \
  --baseline-table outputs/baselines/fairsin/nba_fairsin_baseline_models.csv \
  --candidate-models "${CANDIDATES[@]}" \
  --sample-size 5 \
  --draws 1000 \
  --seed 20260716 \
  --output-prefix outputs/budget_matched_fairness/nba_gcn_vs_fairsin_k5
