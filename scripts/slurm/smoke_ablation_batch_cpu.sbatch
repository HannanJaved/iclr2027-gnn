#!/usr/bin/env bash
#SBATCH --job-name=gnn_ablate_cpu_smoke
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=12G
#SBATCH -p alpha
#SBATCH --requeue
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err
#SBATCH --array=0-1%1

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH=${PROJECT_DIR}/src:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES=""

# Optional: source your local cache/env setup here
PROJECT_DIR=${PROJECT_DIR}
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv/bin/activate"

ARCHITECTURES=(mlp topology_only)
architecture="${ARCHITECTURES[$SLURM_ARRAY_TASK_ID]}"

python -u -m gnn_rashomon.cli.train_batch \
  --dataset cora \
  --model "${architecture}" \
  --sweep seed \
  --seed-start "$((9000 + SLURM_ARRAY_TASK_ID))" \
  --seed-count 1 \
  --max-epochs 1 \
  --device cpu
