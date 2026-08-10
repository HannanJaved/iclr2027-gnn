#!/usr/bin/env bash
#SBATCH --job-name=gnn_pg_xai_smoke
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH -p romeo
#SBATCH --requeue
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err
#SBATCH --array=0-4%2

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
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" .logs

PREFIXES=(
  cora-gcn-seed-relative0.1-epochs200
  cora-gat-arch_gat-relative2-epochs200
  cora-appnp-arch_appnp-relative2-epochs200
  pubmed-gat-arch_gat-relative2-epochs200
  pubmed-appnp-arch_appnp-relative2-epochs200
)

prefix="${PREFIXES[$SLURM_ARRAY_TASK_ID]}"

python -u -m gnn_rashomon.cli.run_explanations \
  --rashomon-set "outputs/rashomon_sets/${prefix}.json" \
  --node-table "outputs/metrics/${prefix}.structure_nodes.csv" \
  --output-prefix "outputs/explanations/${prefix}.pgexplainer-smoke" \
  --max-nodes 5 \
  --max-models 3 \
  --top-k 10 \
  --explainer pgexplainer \
  --explainer-epochs 5 \
  --pgexplainer-lr 0.003 \
  --device cpu
