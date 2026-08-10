#!/usr/bin/env bash
#SBATCH --job-name=gnn_adult_fix_seed
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH -p romeo
#SBATCH --array=0-49%4
#SBATCH --output=.logs/%x_%A_%a.out
#SBATCH --error=.logs/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR}
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/release/adult_corrected}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES=""

module purge
module load release/24.10 GCC/13.2.0 OpenMPI/4.1.6 PyTorch/2.3.0
# Optional: source your local cache/env setup here
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv/bin/activate"

python -u -m gnn_rashomon.cli.train_baseline \
  dataset=adult \
  rashomon_type=baseline \
  seed="${SLURM_ARRAY_TASK_ID}" \
  paths.output_dir="${OUTPUT_ROOT}" \
  trainer.device=cpu \
  trainer.max_epochs=200 \
  dataset_config.source=local \
  dataset_config.csv_path=data/Adult/adult.csv \
  dataset_config.max_nodes=5000 \
  dataset_config.k_neighbors=10 \
  dataset_config.use_cache=true \
  dataset_config.include_sensitive_feature=false
