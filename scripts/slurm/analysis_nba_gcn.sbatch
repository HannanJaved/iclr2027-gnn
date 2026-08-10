#!/usr/bin/env bash
#SBATCH --job-name=gnn_nba_analysis
#SBATCH --time=01:30:00
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
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv-module/bin/activate"

export MPLCONFIGDIR="${TMPDIR:-/tmp}/gnn-rashomon-matplotlib-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/gnn-rashomon-cache-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

for set_type in seed hyperparameter; do
  if [[ "${set_type}" == "seed" ]]; then
    tolerance="relative"
    epsilon="0.10"
    suffix="relative0.1"
    rashomon_type="seed"
    dedupe=()
  else
    tolerance="relative"
    epsilon="2.0"
    suffix="relative2"
    rashomon_type="hyperparameter"
    dedupe=(--no-deduplicate-seeds)
  fi

  python -u -m gnn_rashomon.cli.generate_rashomon \
    --dataset nba \
    --architecture gcn \
    --set-type "${set_type}" \
    --rashomon-type "${rashomon_type}" \
    --max-epochs 200 \
    --tolerance-mode "${tolerance}" \
    --epsilon "${epsilon}" \
    "${dedupe[@]}"

  set_path="outputs/rashomon_sets/nba-gcn-${set_type}-${suffix}-epochs200.json"
  mult_path="outputs/metrics/nba-gcn-${set_type}-${suffix}-epochs200.multiplicity_nodes.csv"
  python -u -m gnn_rashomon.cli.compute_multiplicity \
    "${set_path}" \
    --capacity-method chunked \
    --capacity-chunk-size 16
  python -u -m gnn_rashomon.cli.compute_structure \
    --dataset nba \
    --rashomon-set "${set_path}" \
    --multiplicity-table "${mult_path}" \
    --include-community
  python -u -m gnn_rashomon.cli.run_fairness \
    --dataset nba \
    --rashomon-set "${set_path}" \
    --multiplicity-table "${mult_path}" \
    --sensitive-attribute country
done

python -u -m gnn_rashomon.cli.build_report \
  --set-pattern "nba-gcn-*-relative*-epochs200" \
  --report-name nba_gcn
python -u -m gnn_rashomon.cli.plot_report --report-name nba_gcn
