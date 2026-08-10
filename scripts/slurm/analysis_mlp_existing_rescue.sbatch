#!/usr/bin/env bash
#SBATCH --job-name=gnn_mlp_rescue
#SBATCH --time=01:30:00
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

analyze_set() {
  local dataset="$1"
  local sweep="$2"
  local epsilon="$3"
  local suffix="$4"

  echo "Analyzing existing MLP ${sweep} dataset=${dataset} epsilon=${epsilon}"
  python -u -m gnn_rashomon.cli.generate_rashomon \
    --dataset "${dataset}" \
    --architecture mlp \
    --set-type "${sweep}_rescue" \
    --rashomon-type "${sweep}" \
    --max-epochs 200 \
    --tolerance-mode relative \
    --epsilon "${epsilon}" \
    ${DEDUP_FLAG:-}

  local set_path="outputs/rashomon_sets/${dataset}-mlp-${sweep}_rescue-${suffix}-epochs200.json"
  local mult_path="outputs/metrics/${dataset}-mlp-${sweep}_rescue-${suffix}-epochs200.multiplicity_nodes.csv"
  python -u -m gnn_rashomon.cli.compute_multiplicity \
    "${set_path}" \
    --capacity-method chunked \
    --capacity-chunk-size 16
  python -u -m gnn_rashomon.cli.compute_structure \
    --dataset "${dataset}" \
    --rashomon-set "${set_path}" \
    --multiplicity-table "${mult_path}"
}

# Existing seed MLP sets become useful with a looser, explicitly labeled tolerance.
DEDUP_FLAG=""
analyze_set cora seed 1.0 relative1
analyze_set pubmed seed 1.0 relative1
analyze_set citeseer seed 10.0 relative10

# Existing hyperparameter MLP sets are still sparse for CiteSeer/PubMed, but these
# give a baseline while the tuned hyperparameter rerun trains.
DEDUP_FLAG="--no-deduplicate-seeds"
analyze_set cora hyperparameter 3.0 relative3
analyze_set pubmed hyperparameter 10.0 relative10
analyze_set citeseer hyperparameter 10.0 relative10

python -u -m gnn_rashomon.cli.build_report \
  --set-pattern "*-mlp-seed_rescue-relative1-epochs200" \
  --report-name mlp_seed_rescue_relative1
python -u -m gnn_rashomon.cli.plot_report \
  --report-name mlp_seed_rescue_relative1

python -u -m gnn_rashomon.cli.build_report \
  --set-pattern "citeseer-mlp-seed_rescue-relative10-epochs200" \
  --report-name mlp_seed_rescue_citeseer_relative10
python -u -m gnn_rashomon.cli.plot_report \
  --report-name mlp_seed_rescue_citeseer_relative10

python -u -m gnn_rashomon.cli.build_report \
  --set-pattern "*-mlp-hyperparameter_rescue-relative*-epochs200" \
  --report-name mlp_hyperparameter_rescue
python -u -m gnn_rashomon.cli.plot_report \
  --report-name mlp_hyperparameter_rescue
