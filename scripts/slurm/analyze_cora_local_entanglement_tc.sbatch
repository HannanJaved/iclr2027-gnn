#!/usr/bin/env bash
#SBATCH --job-name=gnn_cora_lent_tc
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=8G
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

python -u -m gnn_rashomon.cli.analyze_local_entanglement \
  --rewiring-metadata outputs/graphs/cora-local_entanglement-tau0p1-0p05-seed0.json \
  --original-structure outputs/metrics/cora-gcn-seed-relative0.1-epochs200.structure_nodes.csv \
  --rewired-structure outputs/metrics/cora-gcn-local_entanglement_tau0p1-relative2-epochs200.structure_nodes.csv \
  --delta-table outputs/metrics/cora-local-entanglement-tau0p1-relative2-seed0.delta_multiplicity.csv \
  --output-prefix outputs/figures/cora_local_entanglement_treatment_control
