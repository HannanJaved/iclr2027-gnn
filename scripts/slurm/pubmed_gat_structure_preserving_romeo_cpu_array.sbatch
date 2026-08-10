#!/usr/bin/env bash
#SBATCH --job-name=gnn_gat_struct
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH -p romeo
#SBATCH --requeue
#SBATCH --array=0-19%8
#SBATCH --output=.logs/%x_%A_%a.out
#SBATCH --error=.logs/%x_%A_%a.err

# PubMed GAT structure-preserving control: reuse existing PubMed SP graphs
# (same topology as GCN control) and evaluate the canonical GAT ensemble.

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

seed="${SLURM_ARRAY_TASK_ID}"
graph_id="pubmed-structure_preserving-0p25-seed${seed}"
src_root="outputs/repeated_rewiring/structure_preserving"
out_root="outputs/non_gcn_rewiring/pubmed_gat/structure_preserving"
set_id="pubmed-gat-canonical_seed-relative2-epochs200-valaccdelta0p02"

mkdir -p "${out_root}/realizations" .logs

python -u -m gnn_rashomon.cli.evaluate_fixed_set_rewiring \
  --rashomon-set "outputs/canonical_pubmed_gat/performance_sensitivity/rashomon_sets/${set_id}.json" \
  --original-node-table "outputs/non_gcn_rewiring/pubmed_gat/metrics/${set_id}.multiplicity_nodes.csv" \
  --graph-path "${src_root}/graphs/${graph_id}.pt" \
  --rewiring-metadata "${src_root}/graphs/${graph_id}.json" \
  --output "${out_root}/realizations/${graph_id}.json" \
  --device cpu
