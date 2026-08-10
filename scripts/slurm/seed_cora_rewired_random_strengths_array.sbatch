#!/usr/bin/env bash
#SBATCH --job-name=gnn_cora_rw_seed
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
#SBATCH --array=0-149%10

set -euo pipefail
export PYTHONUNBUFFERED=1

module load CUDA
# Optional: source your local cache/env setup here
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv/bin/activate"

STRENGTH_IDS=(0p01 0p1 0p25)
strength_idx=$((SLURM_ARRAY_TASK_ID / 50))
seed=$((SLURM_ARRAY_TASK_ID % 50))
strength_id="${STRENGTH_IDS[$strength_idx]}"
GRAPH_PATH="${PROJECT_DIR}/outputs/graphs/cora-random-${strength_id}-seed0.pt"

test -f "${GRAPH_PATH}"

python -u -m gnn_rashomon.cli.train_baseline \
  dataset=cora \
  graph_path="${GRAPH_PATH}" \
  rashomon_type="rewired_random_${strength_id}" \
  seed="${seed}" \
  trainer.device=cuda \
  trainer.max_epochs=200
