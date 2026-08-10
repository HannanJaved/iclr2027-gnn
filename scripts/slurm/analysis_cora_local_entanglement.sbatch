#!/usr/bin/env bash
#SBATCH --job-name=gnn_cora_lent_analysis
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH -p romeo
#SBATCH --requeue
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH=${PROJECT_DIR}/src:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES=""

module purge
module load release/24.10 GCC/13.2.0 OpenMPI/4.1.6 PyTorch/2.3.0

# Optional: source your local cache/env setup here
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv-module/bin/activate"

export MPLCONFIGDIR="${TMPDIR:-/tmp}/gnn-rashomon-matplotlib-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/gnn-rashomon-cache-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

GRAPH_PATH="${PROJECT_DIR}/outputs/graphs/cora-local_entanglement-tau0p1-0p05-seed0.pt"
SET_PATH="outputs/rashomon_sets/cora-gcn-local_entanglement_tau0p1-relative2-epochs200.json"
MULT_TABLE="outputs/metrics/cora-gcn-local_entanglement_tau0p1-relative2-epochs200.multiplicity_nodes.csv"

test -f "${GRAPH_PATH}"

python -u -m gnn_rashomon.cli.generate_rashomon \
  --dataset cora \
  --architecture gcn \
  --set-type "local_entanglement_tau0p1" \
  --rashomon-type "local_entanglement_tau0p1" \
  --max-epochs 200 \
  --tolerance-mode relative \
  --epsilon 2.0

python -u -m gnn_rashomon.cli.compute_multiplicity "${SET_PATH}"

python -u -m gnn_rashomon.cli.compute_structure \
  --dataset cora \
  --rashomon-set "${SET_PATH}" \
  --multiplicity-table "${MULT_TABLE}" \
  --graph-path "${GRAPH_PATH}" \
  --include-community

python -u -m gnn_rashomon.cli.compare_rewired \
  --original-table outputs/metrics/cora-gcn-seed-relative0.1-epochs200.multiplicity_nodes.csv \
  --rewired-table "${MULT_TABLE}" \
  --output "outputs/metrics/cora-local-entanglement-tau0p1-relative2-seed0.delta_multiplicity.csv"

python -u -m gnn_rashomon.cli.build_rewiring_report \
  --dataset cora \
  --mode local_entanglement \
  --strength-ids tau0p1 \
  --strength-values 0.10 \
  --graph-ids local_entanglement-tau0p1-0p05-seed0 \
  --metric-ids local_entanglement_tau0p1 \
  --delta-ids local-entanglement-tau0p1-relative2-seed0 \
  --set-suffix relative2-epochs200 \
  --plot-prefix cora_local_entanglement_rewiring \
  --x-label "Treatment entropy target tau"
