#!/usr/bin/env bash
#SBATCH --job-name=gnn_set_size
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH -p romeo
#SBATCH --requeue
#SBATCH --array=0-21%6
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

SETS=(
  outputs/rashomon_sets/cora-gcn-seed-relative0.1-epochs200.json
  outputs/rashomon_sets/cora-gcn-hyperparameter-relative2-epochs200.json
  outputs/rashomon_sets/citeseer-gcn-seed-relative0.1-epochs200.json
  outputs/rashomon_sets/citeseer-gcn-hyperparameter-relative2-epochs200.json
  outputs/rashomon_sets/pubmed-gcn-seed-relative0.1-epochs200.json
  outputs/rashomon_sets/pubmed-gcn-hyperparameter-relative2-epochs200.json
  outputs/rashomon_sets/cora-gat-arch_gat-relative2-epochs200.json
  outputs/rashomon_sets/cora-appnp-arch_appnp-relative2-epochs200.json
  outputs/rashomon_sets/pubmed-gat-arch_gat-relative2-epochs200.json
  outputs/rashomon_sets/pubmed-graphsage-arch_graphsage-relative2-epochs200.json
  outputs/rashomon_sets/pubmed-appnp-arch_appnp-relative2-epochs200.json
  outputs/rashomon_sets/amazon_photo-gcn-seed-relative0.1-epochs200.json
  outputs/rashomon_sets/amazon_photo-gcn-hyperparameter-relative2-epochs200.json
  outputs/rashomon_sets/amazon_photo-gat-arch_gat-relative0.1-epochs200.json
  outputs/rashomon_sets/amazon_photo-graphsage-arch_graphsage-relative0.1-epochs200.json
  outputs/rashomon_sets/adult-gcn-seed-relative0.1-epochs200.json
  outputs/rashomon_sets/adult-gcn-hyperparameter-relative2-epochs200.json
  outputs/rashomon_sets/german_credit-gcn-seed-relative0.1-epochs200.json
  outputs/rashomon_sets/german_credit-gcn-hyperparameter-relative2-epochs200.json
  outputs/rashomon_sets/nba-gcn-seed-relative0.1-epochs200.json
  outputs/rashomon_sets/nba-gcn-hyperparameter-relative2-epochs200.json
  outputs/rashomon_sets/nba-gat-seed-relative0.1-epochs300.json
)

rashomon_set="${SETS[$SLURM_ARRAY_TASK_ID]}"
set_name="$(basename "${rashomon_set}" .json)"

python -u -m gnn_rashomon.cli.analyze_set_size_robustness \
  --rashomon-set "${rashomon_set}" \
  --sample-size 10 \
  --draws 500 \
  --seed 20260716 \
  --output-prefix "outputs/set_size_robustness/${set_name}-k10"
