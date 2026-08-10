#!/usr/bin/env bash
#SBATCH --job-name=gnn_ablate_seed
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH -p alpha
#SBATCH --requeue
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err
#SBATCH --array=0-299%18

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH=${PROJECT_DIR}/src:${PYTHONPATH:-}

module load CUDA
# Optional: source your local cache/env setup here
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv/bin/activate"

DATASETS=(cora pubmed citeseer)
ARCHITECTURES=(mlp topology_only)

idx="${SLURM_ARRAY_TASK_ID}"
seed=$((idx % 50))
idx=$((idx / 50))
arch_idx=$((idx % ${#ARCHITECTURES[@]}))
idx=$((idx / ${#ARCHITECTURES[@]}))
dataset_idx=$((idx % ${#DATASETS[@]}))

dataset="${DATASETS[$dataset_idx]}"
architecture="${ARCHITECTURES[$arch_idx]}"

echo "Training ablation seed run dataset=${dataset} architecture=${architecture} seed=${seed}"

python -u -m gnn_rashomon.cli.train_baseline \
  dataset="${dataset}" \
  model="${architecture}" \
  rashomon_type=seed \
  seed="${seed}" \
  trainer.device=cuda \
  trainer.max_epochs=200
