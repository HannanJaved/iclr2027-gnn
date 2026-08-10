#!/usr/bin/env bash
#SBATCH --job-name=gnn_cora_hrewire
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
#SBATCH --array=0-1

set -euo pipefail
export PYTHONUNBUFFERED=1

module load CUDA
# Optional: source your local cache/env setup here
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv/bin/activate"

TARGETS=(0.70 0.60)
target="${TARGETS[$SLURM_ARRAY_TASK_ID]}"

python -u -m gnn_rashomon.cli.run_rewiring \
  --dataset cora \
  --mode homophily_targeted \
  --target-homophily "${target}" \
  --tolerance 0.02 \
  --max-attempts 75000 \
  --seed 0
