#!/usr/bin/env bash
#SBATCH --job-name=gnn_nba_gat_audit
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH -p alpha
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH=${PROJECT_DIR}/src:${PYTHONPATH:-}
module purge
module load release/24.10 GCC/13.2.0 OpenMPI/4.1.6 PyTorch/2.3.0
# Optional: source your local cache/env setup here
PROJECT_DIR=${PROJECT_DIR}
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv/bin/activate"

python -u -m gnn_rashomon.cli.audit_rashomon_health \
  --dataset nba --architecture gat --rashomon-type hyperparameter --max-epochs 200 \
  --expected-candidates 36 \
  --tolerance-mode relative --epsilon 0.01 0.05 0.1 0.25 0.5 1.0 2.0 \
  --output-prefix outputs/figures/nba_gat_existing_tolerance_health
