#!/usr/bin/env bash
#SBATCH --job-name=gnn_xai_arch_smoke
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH -p romeo
#SBATCH --requeue
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err
#SBATCH --array=0-3%2

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

DATASETS=(cora cora pubmed pubmed)
ARCHITECTURES=(gat appnp gat appnp)

dataset="${DATASETS[$SLURM_ARRAY_TASK_ID]}"
architecture="${ARCHITECTURES[$SLURM_ARRAY_TASK_ID]}"
prefix="${dataset}-${architecture}-arch_${architecture}-relative2-epochs200"

python -u -m gnn_rashomon.cli.run_explanations \
  --rashomon-set "outputs/rashomon_sets/${prefix}.json" \
  --node-table "outputs/metrics/${prefix}.structure_nodes.csv" \
  --output-prefix "outputs/explanations/${prefix}.gnnexplainer-smoke" \
  --max-nodes 5 \
  --max-models 3 \
  --top-k 10 \
  --explainer-epochs 20 \
  --device cpu
