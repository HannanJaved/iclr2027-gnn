#!/usr/bin/env bash
#SBATCH --job-name=gnn_repeat_local
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH -p romeo
#SBATCH --requeue
#SBATCH --array=0-59%6
#SBATCH --output=.logs/%x_%A_%a.out
#SBATCH --error=.logs/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
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
dataset_idx=$((SLURM_ARRAY_TASK_ID / 20))
seed=$((SLURM_ARRAY_TASK_ID % 20))
dataset="${DATASETS[$dataset_idx]}"
graph_id="${dataset}-local_entanglement-tau0p1-0p05-seed${seed}"
output_root="outputs/repeated_rewiring/local"

python -u -m gnn_rashomon.cli.run_rewiring \
  --dataset "${dataset}" \
  --mode local_entanglement \
  --strength 0.05 \
  --tau 0.10 \
  --delta-e 0.02 \
  --max-feature-drift 0.02 \
  --treatment-count 50 \
  --min-degree 2 \
  --max-attempts 50000 \
  --seed "${seed}" \
  --output-dir "${output_root}"

python -u -m gnn_rashomon.cli.evaluate_fixed_set_rewiring \
  --rashomon-set "outputs/rashomon_sets/${dataset}-gcn-seed-relative0.1-epochs200.json" \
  --original-node-table "outputs/metrics/${dataset}-gcn-seed-relative0.1-epochs200.multiplicity_nodes.csv" \
  --graph-path "${output_root}/graphs/${graph_id}.pt" \
  --rewiring-metadata "${output_root}/graphs/${graph_id}.json" \
  --output "${output_root}/realizations/${graph_id}.json"
