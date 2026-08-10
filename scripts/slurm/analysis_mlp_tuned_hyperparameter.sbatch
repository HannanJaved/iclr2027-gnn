#!/usr/bin/env bash
#SBATCH --job-name=gnn_mlp_tuned_analysis
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
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

analyze_tuned() {
  local dataset="$1"
  local epsilon="$2"
  local suffix="$3"

  echo "Analyzing tuned MLP hyperparameter dataset=${dataset} epsilon=${epsilon}"
  python -u -m gnn_rashomon.cli.generate_rashomon \
    --dataset "${dataset}" \
    --architecture mlp \
    --set-type hyperparameter_tuned \
    --rashomon-type hyperparameter_tuned \
    --max-epochs 200 \
    --tolerance-mode relative \
    --epsilon "${epsilon}" \
    --no-deduplicate-seeds

  local set_path="outputs/rashomon_sets/${dataset}-mlp-hyperparameter_tuned-${suffix}-epochs200.json"
  local mult_path="outputs/metrics/${dataset}-mlp-hyperparameter_tuned-${suffix}-epochs200.multiplicity_nodes.csv"
  python -u -m gnn_rashomon.cli.compute_multiplicity \
    "${set_path}" \
    --capacity-method chunked \
    --capacity-chunk-size 16
  python -u -m gnn_rashomon.cli.compute_structure \
    --dataset "${dataset}" \
    --rashomon-set "${set_path}" \
    --multiplicity-table "${mult_path}"
}

for dataset in cora citeseer pubmed; do
  analyze_tuned "${dataset}" 2.0 relative2
  analyze_tuned "${dataset}" 5.0 relative5
  analyze_tuned "${dataset}" 10.0 relative10
done

for suffix in relative2 relative5 relative10; do
  python -u -m gnn_rashomon.cli.build_report \
    --set-pattern "*-mlp-hyperparameter_tuned-${suffix}-epochs200" \
    --report-name "mlp_hyperparameter_tuned_${suffix}"
  python -u -m gnn_rashomon.cli.plot_report \
    --report-name "mlp_hyperparameter_tuned_${suffix}"
done
