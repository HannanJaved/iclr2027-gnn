#!/usr/bin/env bash
#SBATCH --job-name=gnn_cora_rw_report
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH -p alpha
#SBATCH --requeue
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

module load CUDA
# Optional: source your local cache/env setup here
PROJECT_DIR=${PROJECT_DIR}
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv/bin/activate"

python -m pip install -e .

python -u -m gnn_rashomon.cli.build_rewiring_report \
  --dataset cora \
  --mode random \
  --strength-ids 0p01 0p05 0p1 0p25 \
  --strength-values 0.01 0.05 0.10 0.25
