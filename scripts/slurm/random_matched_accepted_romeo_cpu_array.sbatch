#!/usr/bin/env bash
#SBATCH --job-name=gnn_rand_matched
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH -p romeo
#SBATCH --requeue
#SBATCH --array=0-59%12
#SBATCH --output=.logs/%x_%A_%a.out
#SBATCH --error=.logs/%x_%A_%a.err

# Unconstrained random rewiring stopped at the structure-preserving accepted-swap
# count (mean over the SP runs): CiteSeer 131, Cora 142, PubMed 2792.
# 3 datasets x 20 seeds = 60 tasks.

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

DATASETS=(cora citeseer pubmed)
# Mean accepted swaps from structure_preserving 0.25 runs
TARGETS=(142 131 2792)
dataset_idx=$((SLURM_ARRAY_TASK_ID / 20))
seed=$((SLURM_ARRAY_TASK_ID % 20))
dataset="${DATASETS[$dataset_idx]}"
target="${TARGETS[$dataset_idx]}"
graph_id="${dataset}-random_matched_accepted-acc${target}-matched-seed${seed}"
output_root="outputs/repeated_rewiring/random_matched_accepted"
set_id="${dataset}-gcn-seed-relative0.1-epochs200"

mkdir -p "${output_root}/graphs" "${output_root}/realizations" .logs

python -u -m gnn_rashomon.cli.run_rewiring \
  --dataset "${dataset}" \
  --mode random \
  --strength 0.0 \
  --target-accepted-swaps "${target}" \
  --seed "${seed}" \
  --output-dir "${output_root}"

python -u -m gnn_rashomon.cli.evaluate_fixed_set_rewiring \
  --rashomon-set "outputs/rashomon_sets/${set_id}.json" \
  --original-node-table "outputs/metrics/${set_id}.multiplicity_nodes.csv" \
  --graph-path "${output_root}/graphs/${graph_id}.pt" \
  --rewiring-metadata "${output_root}/graphs/${graph_id}.json" \
  --output "${output_root}/realizations/${graph_id}.json" \
  --device cpu
