#!/usr/bin/env bash
#SBATCH --job-name=gnn_ogbn_seed
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -p alpha
#SBATCH --array=0-49%4
#SBATCH --requeue
#SBATCH --output=.logs/%x_%A_%a.out
#SBATCH --error=.logs/%x_%A_%a.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH=${PROJECT_DIR}/src:${PROJECT_DIR}/.alpha-deps:${PYTHONPATH:-}
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
module purge
module load release/24.10 GCC/13.2.0 OpenMPI/4.1.6 PyTorch/2.3.0
# Optional: source your local cache/env setup here
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv/bin/activate"
mkdir -p .logs
python -u -m gnn_rashomon.cli.train_baseline \
  dataset=ogbn_arxiv \
  model=gcn \
  rashomon_type=seed_ogbn_revised \
  seed="${SLURM_ARRAY_TASK_ID}" \
  dataset_config.normalize_features=false \
  trainer.device=cuda \
  trainer.max_epochs=300 \
  trainer.early_stopping_patience=100 \
  trainer.learning_rate=0.01 \
  trainer.weight_decay=0.0005 \
  model_config.hidden_channels=256 \
  model_config.dropout=0.5
