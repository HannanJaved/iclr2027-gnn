#!/usr/bin/env bash
#SBATCH --job-name=gnn_ablate_smoke
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH -p alpha
#SBATCH --requeue
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err
#SBATCH --array=0-1%2

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH=${PROJECT_DIR}/src:${PYTHONPATH:-}

module load CUDA
# Optional: source your local cache/env setup here
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv/bin/activate"

ARCHITECTURES=(mlp topology_only)
architecture="${ARCHITECTURES[$SLURM_ARRAY_TASK_ID]}"

echo "Starting ablation smoke architecture=${architecture} at $(date)"

python -u -m gnn_rashomon.cli.train_baseline \
  dataset=cora \
  model="${architecture}" \
  rashomon_type=ablation_smoke \
  seed="$((9000 + SLURM_ARRAY_TASK_ID))" \
  trainer.device=cuda \
  trainer.max_epochs=1

echo "Finished ablation smoke architecture=${architecture} at $(date)"
