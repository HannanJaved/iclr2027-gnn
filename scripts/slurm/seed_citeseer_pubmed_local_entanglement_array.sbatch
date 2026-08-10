#!/usr/bin/env bash
#SBATCH --job-name=gnn_cp_lent_seed
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH -p romeo
#SBATCH --requeue
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err
#SBATCH --array=0-99%8

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

DATASETS=(citeseer pubmed)
dataset_idx=$((SLURM_ARRAY_TASK_ID / 50))
seed=$((SLURM_ARRAY_TASK_ID % 50))
dataset="${DATASETS[$dataset_idx]}"
graph_path="${PROJECT_DIR}/outputs/graphs/${dataset}-local_entanglement-tau0p1-0p05-seed0.pt"

test -f "${graph_path}"

python -u -m gnn_rashomon.cli.train_baseline \
  dataset="${dataset}" \
  graph_path="${graph_path}" \
  rashomon_type="local_entanglement_tau0p1" \
  seed="${seed}" \
  trainer.device=cpu \
  trainer.max_epochs=200
