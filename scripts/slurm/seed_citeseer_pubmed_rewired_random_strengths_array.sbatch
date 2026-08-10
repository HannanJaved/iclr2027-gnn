#!/usr/bin/env bash
#SBATCH --job-name=gnn_cp_rw_seed
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH -p romeo
#SBATCH --requeue
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err
#SBATCH --array=0-399%8

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

DATASETS=(citeseer pubmed)
STRENGTH_IDS=(0p01 0p05 0p1 0p25)

dataset_idx=$((SLURM_ARRAY_TASK_ID / 200))
within_dataset=$((SLURM_ARRAY_TASK_ID % 200))
strength_idx=$((within_dataset / 50))
seed=$((within_dataset % 50))
dataset="${DATASETS[$dataset_idx]}"
strength_id="${STRENGTH_IDS[$strength_idx]}"
graph_path="${PROJECT_DIR}/outputs/graphs/${dataset}-random-${strength_id}-seed0.pt"

test -f "${graph_path}"

python -u -m gnn_rashomon.cli.train_baseline \
  dataset="${dataset}" \
  graph_path="${graph_path}" \
  rashomon_type="rewired_random_${strength_id}" \
  seed="${seed}" \
  trainer.device=cpu \
  trainer.max_epochs=200
