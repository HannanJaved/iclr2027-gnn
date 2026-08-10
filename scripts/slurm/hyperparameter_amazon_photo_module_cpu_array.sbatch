#!/usr/bin/env bash
#SBATCH --job-name=gnn_amzphoto_hp
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH -p romeo
#SBATCH --requeue
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err
#SBATCH --array=0-35%4

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH=${PROJECT_DIR}/src:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES=""

module purge
module load release/24.10 GCC/13.2.0 OpenMPI/4.1.6 PyTorch/2.3.0

# Optional: source your local cache/env setup here
PROJECT_DIR=${PROJECT_DIR}
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv-module/bin/activate"

LEARNING_RATES=(0.02 0.01 0.005)
DROPOUTS=(0.3 0.5 0.7)
WEIGHT_DECAYS=(0.0005 0.0001)
HIDDEN_CHANNELS=(16 32)

idx="${SLURM_ARRAY_TASK_ID}"
hidden_idx=$((idx % ${#HIDDEN_CHANNELS[@]}))
idx=$((idx / ${#HIDDEN_CHANNELS[@]}))
wd_idx=$((idx % ${#WEIGHT_DECAYS[@]}))
idx=$((idx / ${#WEIGHT_DECAYS[@]}))
dropout_idx=$((idx % ${#DROPOUTS[@]}))
idx=$((idx / ${#DROPOUTS[@]}))
lr_idx=$((idx % ${#LEARNING_RATES[@]}))

python -u -m gnn_rashomon.cli.train_baseline \
  dataset=amazon_photo \
  model=gcn \
  rashomon_type=hyperparameter \
  seed=0 \
  trainer.device=cpu \
  trainer.max_epochs=200 \
  trainer.learning_rate="${LEARNING_RATES[$lr_idx]}" \
  trainer.weight_decay="${WEIGHT_DECAYS[$wd_idx]}" \
  model_config.dropout="${DROPOUTS[$dropout_idx]}" \
  model_config.hidden_channels="${HIDDEN_CHANNELS[$hidden_idx]}"
