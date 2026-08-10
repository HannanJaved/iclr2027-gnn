#!/usr/bin/env bash
#SBATCH --job-name=gnn_cora_lent_seed
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=12G
#SBATCH -p romeo
#SBATCH --requeue
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err
#SBATCH --array=0-49%10

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

GRAPH_PATH="${PROJECT_DIR}/outputs/graphs/cora-local_entanglement-tau0p1-0p05-seed0.pt"
test -f "${GRAPH_PATH}"

python -u -m gnn_rashomon.cli.train_baseline \
  dataset=cora \
  graph_path="${GRAPH_PATH}" \
  rashomon_type="local_entanglement_tau0p1" \
  seed="${SLURM_ARRAY_TASK_ID}" \
  trainer.device=cpu \
  trainer.max_epochs=200
