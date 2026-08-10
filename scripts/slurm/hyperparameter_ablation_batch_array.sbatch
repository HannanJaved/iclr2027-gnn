#!/usr/bin/env bash
#SBATCH --job-name=gnn_ablate_hp_b
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -p alpha
#SBATCH --requeue
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err
#SBATCH --array=0-5%2

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH=${PROJECT_DIR}/src:${PYTHONPATH:-}

module load CUDA
# Optional: source your local cache/env setup here
PROJECT_DIR=${PROJECT_DIR}
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv/bin/activate"

DATASETS=(cora pubmed citeseer)
ARCHITECTURES=(mlp topology_only)
idx="${SLURM_ARRAY_TASK_ID}"
arch_idx=$((idx % ${#ARCHITECTURES[@]}))
dataset_idx=$((idx / ${#ARCHITECTURES[@]}))

python -u -m gnn_rashomon.cli.train_batch \
  --dataset "${DATASETS[$dataset_idx]}" \
  --model "${ARCHITECTURES[$arch_idx]}" \
  --sweep hyperparameter \
  --max-epochs 200 \
  --device cuda
