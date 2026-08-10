#!/usr/bin/env bash
#SBATCH --job-name=gnn_arch_seed
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH -p romeo
#SBATCH --requeue
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err
#SBATCH --array=0-299%8

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

DATASETS=(cora pubmed)
ARCHITECTURES=(gat graphsage appnp)

dataset_idx=$((SLURM_ARRAY_TASK_ID / 150))
within_dataset=$((SLURM_ARRAY_TASK_ID % 150))
arch_idx=$((within_dataset / 50))
seed=$((within_dataset % 50))

dataset="${DATASETS[$dataset_idx]}"
architecture="${ARCHITECTURES[$arch_idx]}"

python -u -m gnn_rashomon.cli.train_baseline \
  dataset="${dataset}" \
  model="${architecture}" \
  rashomon_type="arch_${architecture}" \
  seed="${seed}" \
  trainer.device=cpu \
  trainer.max_epochs=200
