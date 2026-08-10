#!/usr/bin/env bash
#SBATCH --job-name=gnn_amzphoto_hp_analysis
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
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
PROJECT_DIR=${PROJECT_DIR}
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv-module/bin/activate"

export MPLCONFIGDIR="${TMPDIR:-/tmp}/gnn-rashomon-matplotlib-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/gnn-rashomon-cache-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

python -u -m gnn_rashomon.cli.generate_rashomon \
  --dataset amazon_photo \
  --architecture gcn \
  --set-type hyperparameter \
  --rashomon-type hyperparameter \
  --max-epochs 200 \
  --tolerance-mode relative \
  --epsilon 2.0 \
  --no-deduplicate-seeds

set_path="outputs/rashomon_sets/amazon_photo-gcn-hyperparameter-relative2-epochs200.json"
mult_path="outputs/metrics/amazon_photo-gcn-hyperparameter-relative2-epochs200.multiplicity_nodes.csv"

python -u -m gnn_rashomon.cli.compute_multiplicity \
  "${set_path}" \
  --capacity-method chunked \
  --capacity-chunk-size 16

python -u -m gnn_rashomon.cli.compute_structure \
  --dataset amazon_photo \
  --rashomon-set "${set_path}" \
  --multiplicity-table "${mult_path}" \
  --include-community

python -u -m gnn_rashomon.cli.build_report \
  --set-pattern "amazon_photo-gcn-*-relative*-epochs200" \
  --report-name amazon_photo_gcn
python -u -m gnn_rashomon.cli.plot_report --report-name amazon_photo_gcn
