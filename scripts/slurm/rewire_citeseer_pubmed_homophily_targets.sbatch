#!/usr/bin/env bash
#SBATCH --job-name=gnn_cp_hrew_graph
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH -p romeo
#SBATCH --requeue
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err
#SBATCH --array=0-3

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
TARGETS=(0.70 0.60)

dataset_idx=$((SLURM_ARRAY_TASK_ID / ${#TARGETS[@]}))
target_idx=$((SLURM_ARRAY_TASK_ID % ${#TARGETS[@]}))
dataset="${DATASETS[$dataset_idx]}"
target="${TARGETS[$target_idx]}"

python -u -m gnn_rashomon.cli.run_rewiring \
  --dataset "${dataset}" \
  --mode homophily_targeted \
  --target-homophily "${target}" \
  --tolerance 0.02 \
  --max-attempts 75000 \
  --seed 0
