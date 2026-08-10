#!/usr/bin/env bash
#SBATCH --job-name=gnn_nba_gat_hp
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -p alpha
#SBATCH --array=0-35%6
#SBATCH --requeue
#SBATCH --output=.logs/%x_%A_%a.out
#SBATCH --error=.logs/%x_%A_%a.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH=${PROJECT_DIR}/src:${PYTHONPATH:-}
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
module purge
module load release/24.10 GCC/13.2.0 OpenMPI/4.1.6 PyTorch/2.3.0
# Optional: source your local cache/env setup here
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv/bin/activate"

LEARNING_RATES=(0.0005 0.001 0.003)
DROPOUTS=(0.1 0.3)
HEADS=(1 2)
HIDDEN_CHANNELS=(32 64 128)
idx="${SLURM_ARRAY_TASK_ID}"
hidden_idx=$((idx % 3)); idx=$((idx / 3))
heads_idx=$((idx % 2)); idx=$((idx / 2))
dropout_idx=$((idx % 2)); idx=$((idx / 2))
lr_idx=$((idx % 3))
python -u -m gnn_rashomon.cli.train_baseline \
  dataset=nba model=gat rashomon_type=hyperparameter_nba_gat_revised seed=0 \
  trainer.device=cuda trainer.max_epochs=300 trainer.early_stopping_patience=100 \
  trainer.learning_rate="${LEARNING_RATES[$lr_idx]}" \
  trainer.weight_decay=0.0001 \
  model_config.dropout="${DROPOUTS[$dropout_idx]}" \
  model_config.heads="${HEADS[$heads_idx]}" \
  model_config.hidden_channels="${HIDDEN_CHANNELS[$hidden_idx]}" \
  dataset_config.use_cache=true
