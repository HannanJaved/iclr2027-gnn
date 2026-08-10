#!/usr/bin/env bash
#SBATCH --job-name=gat_canon_seed
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH -p romeo
#SBATCH --requeue
#SBATCH --array=0-49%8
#SBATCH --output=.logs/%x_%A_%a.out
#SBATCH --error=.logs/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR}
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES=""

module purge
module load release/24.10 GCC/13.2.0 OpenMPI/4.1.6 PyTorch/2.3.0
# Optional: source your local cache/env setup here
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv-module/bin/activate"

python -u -m gnn_rashomon.cli.train_baseline \
  dataset=pubmed \
  model=gat \
  rashomon_type=canonical_arch_gat \
  seed="${SLURM_ARRAY_TASK_ID}" \
  paths.output_dir=outputs/canonical_pubmed_gat \
  trainer.device=cpu \
  trainer.max_epochs=200
