#!/usr/bin/env bash
#SBATCH --job-name=gat_fixed_rw
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH -p romeo
#SBATCH --requeue
#SBATCH --array=0-139%8
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

condition=$((SLURM_ARRAY_TASK_ID / 20))
seed=$((SLURM_ARRAY_TASK_ID % 20))
if (( condition < 4 )); then
  STRENGTH_IDS=(0p01 0p05 0p1 0p25)
  graph_id="pubmed-random-${STRENGTH_IDS[$condition]}-seed${seed}"
  source_root=outputs/repeated_rewiring/random
elif (( condition < 6 )); then
  TARGET_IDS=(0p72 0p62)
  target_idx=$((condition - 4))
  graph_id="pubmed-homophily_targeted-target${TARGET_IDS[$target_idx]}-0p05-seed${seed}"
  source_root=outputs/repeated_rewiring/homophily
else
  graph_id="pubmed-local_entanglement-tau0p1-0p05-seed${seed}"
  source_root=outputs/repeated_rewiring/local
fi

python -u -m gnn_rashomon.cli.evaluate_fixed_set_rewiring \
  --rashomon-set outputs/canonical_pubmed_gat/performance_sensitivity/rashomon_sets/pubmed-gat-canonical_seed-relative2-epochs200-valaccdelta0p02.json \
  --original-node-table outputs/non_gcn_rewiring/pubmed_gat/metrics/pubmed-gat-canonical_seed-relative2-epochs200-valaccdelta0p02.multiplicity_nodes.csv \
  --graph-path "${source_root}/graphs/${graph_id}.pt" \
  --rewiring-metadata "${source_root}/graphs/${graph_id}.json" \
  --output "outputs/non_gcn_rewiring/pubmed_gat/realizations/${graph_id}.json"
