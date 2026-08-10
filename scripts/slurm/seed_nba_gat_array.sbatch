#!/usr/bin/env bash
#SBATCH --job-name=gnn_nba_gat_seed
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -p alpha
#SBATCH --array=0-49%6
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
PROJECT_DIR=${PROJECT_DIR}
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv/bin/activate"
python -u -m gnn_rashomon.cli.train_baseline \
  dataset=nba model=gat rashomon_type=seed_nba_gat_revised seed="${SLURM_ARRAY_TASK_ID}" \
  dataset_config.use_cache=true model_config.hidden_channels=64 \
  model_config.heads=2 model_config.dropout=0.3 \
  trainer.device=cuda trainer.max_epochs=300 trainer.early_stopping_patience=100 \
  trainer.learning_rate=0.001 trainer.weight_decay=0.0001
