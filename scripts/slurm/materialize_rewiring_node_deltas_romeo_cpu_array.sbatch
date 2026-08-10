#!/usr/bin/env bash
#SBATCH --job-name=rewire_node_delta
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH -p romeo
#SBATCH --requeue
#SBATCH --array=0-239%8
#SBATCH --output=.logs/%x_%A_%a.out
#SBATCH --error=.logs/%x_%A_%a.err

# Optional #4 support: rematerialize node-level ΔD_i for existing random rewiring graphs.
# Same indexing as repeated_random_rewiring_romeo_cpu_array.sbatch (3x4x20=240).

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
STRENGTHS=(0.01 0.05 0.10 0.25)
STRENGTH_IDS=(0p01 0p05 0p1 0p25)
dataset_idx=$((SLURM_ARRAY_TASK_ID / 80))
within_dataset=$((SLURM_ARRAY_TASK_ID % 80))
strength_idx=$((within_dataset / 20))
seed=$((within_dataset % 20))
dataset="${DATASETS[$dataset_idx]}"
strength_id="${STRENGTH_IDS[$strength_idx]}"
graph_id="${dataset}-random-${strength_id}-seed${seed}"
output_root="outputs/repeated_rewiring/random"

python -u -m gnn_rashomon.cli.evaluate_fixed_set_rewiring \
  --rashomon-set "outputs/rashomon_sets/${dataset}-gcn-seed-relative0.1-epochs200.json" \
  --original-node-table "outputs/metrics/${dataset}-gcn-seed-relative0.1-epochs200.multiplicity_nodes.csv" \
  --graph-path "${output_root}/graphs/${graph_id}.pt" \
  --rewiring-metadata "${output_root}/graphs/${graph_id}.json" \
  --output "${output_root}/realizations/${graph_id}.json" \
  --node-output "${output_root}/node_deltas/${graph_id}.csv"
