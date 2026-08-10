#!/usr/bin/env bash
#SBATCH --job-name=gnn_amz_xai
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -p romeo
#SBATCH --requeue
#SBATCH --output=.logs/%x_%A_%a.out
#SBATCH --error=.logs/%x_%A_%a.err
#SBATCH --array=0-2%2

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

export MPLCONFIGDIR="${TMPDIR:-/tmp}/gnn-rashomon-matplotlib-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/gnn-rashomon-cache-${SLURM_JOB_ID}-${SLURM_ARRAY_TASK_ID}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

PREFIXES=(
  "amazon_photo-gcn-seed-relative0.1-epochs200"
  "amazon_photo-gat-arch_gat-relative0.1-epochs200"
  "amazon_photo-graphsage-arch_graphsage-relative0.1-epochs200"
)

prefix="${PREFIXES[$SLURM_ARRAY_TASK_ID]}"

python -u -m gnn_rashomon.cli.run_explanations \
  --dataset amazon_photo \
  --rashomon-set "outputs/rashomon_sets/${prefix}.json" \
  --node-table "outputs/metrics/${prefix}.structure_nodes.csv" \
  --output-prefix "outputs/explanations/${prefix}.gnnexplainer" \
  --max-nodes 30 \
  --max-models 8 \
  --top-k 10 \
  --explainer-epochs 50 \
  --device cpu
