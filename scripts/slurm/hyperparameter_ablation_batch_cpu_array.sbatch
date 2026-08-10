#!/usr/bin/env bash
#SBATCH --job-name=gnn_ablate_hp_cpu
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH -p alpha
#SBATCH --requeue
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err
#SBATCH --array=0-5%2

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH=${PROJECT_DIR}/src:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES=""

# Optional: source your local cache/env setup here
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
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
  --device cpu
