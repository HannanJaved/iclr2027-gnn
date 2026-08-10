#!/usr/bin/env bash
#SBATCH --job-name=gnn_cora_hrew_report
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH -p alpha
#SBATCH --requeue
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH=${PROJECT_DIR}/src:${PYTHONPATH:-}

module load CUDA
# Optional: source your local cache/env setup here
PROJECT_DIR=${PROJECT_DIR}
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv/bin/activate"

export MPLCONFIGDIR="${TMPDIR:-/tmp}/gnn-rashomon-matplotlib-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/gnn-rashomon-cache-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

echo "Building Cora homophily-targeted rewiring report at $(date)"

python -u -m gnn_rashomon.cli.build_rewiring_report \
  --plot-only \
  --dataset cora \
  --mode homophily_targeted \
  --strength-ids target0p7 target0p6 \
  --strength-values 0.70 0.60 \
  --graph-ids homophily_targeted-target0p7-0p05-seed0 homophily_targeted-target0p6-0p05-seed0 \
  --metric-ids homophily_targeted_0p7 homophily_targeted_0p6 \
  --delta-ids homophily-targeted-0p7-seed0 homophily-targeted-0p6-seed0 \
  --plot-prefix cora_homophily_targeted_rewiring \
  --x-label "Target homophily"

echo "Finished Cora homophily-targeted rewiring report at $(date)"
