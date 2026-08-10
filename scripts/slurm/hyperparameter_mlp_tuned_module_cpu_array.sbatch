#!/usr/bin/env bash
#SBATCH --job-name=gnn_mlp_tuned_hp
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH -p romeo
#SBATCH --requeue
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err
#SBATCH --array=0-2%2

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

DATASETS=(cora citeseer pubmed)
dataset="${DATASETS[$SLURM_ARRAY_TASK_ID]}"

case "${dataset}" in
  cora)
    LRS=(0.003 0.005 0.007 0.01 0.02)
    DROPOUTS=(0.2 0.3 0.4 0.5)
    WEIGHT_DECAYS=(0.00005 0.0001 0.0005)
    HIDDEN=(16 32 64)
    ;;
  citeseer)
    LRS=(0.003 0.005 0.007 0.01)
    DROPOUTS=(0.2 0.3 0.4 0.5)
    WEIGHT_DECAYS=(0.0001 0.0005 0.001)
    HIDDEN=(16 32 64)
    ;;
  pubmed)
    LRS=(0.01 0.015 0.02 0.025)
    DROPOUTS=(0.2 0.3 0.5 0.7)
    WEIGHT_DECAYS=(0.0001 0.0005 0.001)
    HIDDEN=(16 32 64)
    ;;
  *)
    echo "Unknown dataset ${dataset}" >&2
    exit 1
    ;;
esac

python -u -m gnn_rashomon.cli.train_batch \
  --dataset "${dataset}" \
  --model mlp \
  --sweep hyperparameter \
  --rashomon-type hyperparameter_tuned \
  --max-epochs 200 \
  --device cpu \
  --learning-rates "${LRS[@]}" \
  --dropouts "${DROPOUTS[@]}" \
  --weight-decays "${WEIGHT_DECAYS[@]}" \
  --hidden-channels "${HIDDEN[@]}"
