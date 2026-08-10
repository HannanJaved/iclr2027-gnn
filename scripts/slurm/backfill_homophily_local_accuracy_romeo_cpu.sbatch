#!/usr/bin/env bash
#SBATCH --job-name=gnn_acc_homloc
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -p romeo
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err

# Re-evaluate fixed ensembles on saved homophily-targeted (0.62) and local
# entropy graphs. No retraining.

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

python -u -m gnn_rashomon.cli.backfill_rewiring_accuracy \
  --modes homophily_targeted,local_entanglement \
  --strengths 0.05 \
  --device cpu \
  --skip-if-done \
  --realizations-csv outputs/repeated_rewiring/repeated_rewiring_realizations.csv \
  --realizations-csv outputs/non_gcn_rewiring/pubmed_gat/pubmed_gat_repeated_rewiring_realizations.csv \
  --output-dir outputs/rewiring_accuracy
