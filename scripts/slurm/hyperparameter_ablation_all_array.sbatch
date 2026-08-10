#!/usr/bin/env bash
#SBATCH --job-name=gnn_ablate_hp
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
#SBATCH --array=0-215%18

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH=${PROJECT_DIR}/src:${PYTHONPATH:-}

module load CUDA
# Optional: source your local cache/env setup here
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv/bin/activate"

DATASETS=(cora pubmed citeseer)
ARCHITECTURES=(mlp topology_only)
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
idx=$((idx / ${#LEARNING_RATES[@]}))
arch_idx=$((idx % ${#ARCHITECTURES[@]}))
idx=$((idx / ${#ARCHITECTURES[@]}))
dataset_idx=$((idx % ${#DATASETS[@]}))

dataset="${DATASETS[$dataset_idx]}"
architecture="${ARCHITECTURES[$arch_idx]}"

echo "Training ablation hyperparameter run dataset=${dataset} architecture=${architecture}"

python -u -m gnn_rashomon.cli.train_baseline \
  dataset="${dataset}" \
  model="${architecture}" \
  rashomon_type=hyperparameter \
  seed=0 \
  trainer.device=cuda \
  trainer.max_epochs=200 \
  trainer.learning_rate="${LEARNING_RATES[$lr_idx]}" \
  trainer.weight_decay="${WEIGHT_DECAYS[$wd_idx]}" \
  model_config.dropout="${DROPOUTS[$dropout_idx]}" \
  model_config.hidden_channels="${HIDDEN_CHANNELS[$hidden_idx]}"
