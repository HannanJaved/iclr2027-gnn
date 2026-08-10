#!/usr/bin/env bash
#SBATCH --job-name=gnn_cora_hrew_retry
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
#SBATCH --array=0-3%4

set -euo pipefail
export PYTHONUNBUFFERED=1

module load CUDA
# Optional: source your local cache/env setup here
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv/bin/activate"

TARGET_IDS=(0p7 0p6 0p6 0p6)
SEEDS=(45 16 32 38)

target_id="${TARGET_IDS[$SLURM_ARRAY_TASK_ID]}"
seed="${SEEDS[$SLURM_ARRAY_TASK_ID]}"
GRAPH_PATH="${PROJECT_DIR}/outputs/graphs/cora-homophily_targeted-target${target_id}-0p05-seed0.pt"

test -f "${GRAPH_PATH}"

echo "Retrying Cora homophily-targeted target=${target_id} seed=${seed}"

python -u -m gnn_rashomon.cli.train_baseline \
  dataset=cora \
  graph_path="${GRAPH_PATH}" \
  rashomon_type="homophily_targeted_${target_id}" \
  seed="${seed}" \
  trainer.device=cuda \
  trainer.max_epochs=200
