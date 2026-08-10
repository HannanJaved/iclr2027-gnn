#!/usr/bin/env bash
#SBATCH --job-name=gnn_plot_reports
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

for report in seed_relative0.1 hyperparameter_relative2; do
  echo "Plotting report ${report}"
  python -u -m gnn_rashomon.cli.plot_report --report-name "${report}"
done

echo "Report plotting complete"
