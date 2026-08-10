#!/usr/bin/env bash
#SBATCH --job-name=gnn_adult_fix_analysis
#SBATCH --time=02:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH -p romeo
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR}
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/release/adult_corrected}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES=""
export MPLCONFIGDIR="${TMPDIR:-/tmp}/gnn-rashomon-matplotlib-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/gnn-rashomon-cache-${SLURM_JOB_ID}"

module purge
module load release/24.10 GCC/13.2.0 OpenMPI/4.1.6 PyTorch/2.3.0
# Optional: source your local cache/env setup here
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv/bin/activate"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}" "${OUTPUT_ROOT}/figures"

seed_records=$(find "${OUTPUT_ROOT}/metrics" -maxdepth 1 -type f -name 'adult-gcn-*.json' -exec grep -l '"rashomon_type": "baseline"' {} + | wc -l)
hp_records=$(find "${OUTPUT_ROOT}/metrics" -maxdepth 1 -type f -name 'adult-gcn-*.json' -exec grep -l '"rashomon_type": "hyperparameter"' {} + | wc -l)
[[ "${seed_records}" -eq 50 ]] || { echo "Expected 50 seed records, found ${seed_records}" >&2; exit 1; }
[[ "${hp_records}" -eq 36 ]] || { echo "Expected 36 hyperparameter records, found ${hp_records}" >&2; exit 1; }

for set_type in seed hyperparameter; do
  if [[ "${set_type}" == "seed" ]]; then
    epsilon="0.10"
    suffix="relative0.1"
    rashomon_type="baseline"
    dedupe=()
  else
    epsilon="2.0"
    suffix="relative2"
    rashomon_type="hyperparameter"
    dedupe=(--no-deduplicate-seeds)
  fi

  python -u -m gnn_rashomon.cli.generate_rashomon \
    --dataset adult \
    --architecture gcn \
    --set-type "${set_type}" \
    --rashomon-type "${rashomon_type}" \
    --output-dir "${OUTPUT_ROOT}" \
    --max-epochs 200 \
    --tolerance-mode relative \
    --epsilon "${epsilon}" \
    "${dedupe[@]}"

  set_path="${OUTPUT_ROOT}/rashomon_sets/adult-gcn-${set_type}-${suffix}-epochs200.json"
  mult_path="${OUTPUT_ROOT}/metrics/adult-gcn-${set_type}-${suffix}-epochs200.multiplicity_nodes.csv"

  python -u -m gnn_rashomon.cli.compute_multiplicity \
    "${set_path}" \
    --output-dir "${OUTPUT_ROOT}" \
    --capacity-method chunked \
    --capacity-chunk-size 16
  python -u -m gnn_rashomon.cli.compute_structure \
    --dataset adult \
    --rashomon-set "${set_path}" \
    --multiplicity-table "${mult_path}" \
    --include-community \
    --output-dir "${OUTPUT_ROOT}"
  python -u -m gnn_rashomon.cli.run_fairness \
    --dataset adult \
    --rashomon-set "${set_path}" \
    --multiplicity-table "${mult_path}" \
    --sensitive-attribute sex \
    --output-dir "${OUTPUT_ROOT}"

  for metric in demographic_parity_gap equalized_odds_gap degree_disparity; do
    python -u -m gnn_rashomon.cli.build_pareto \
      --mode fairness \
      --fairness-table "${OUTPUT_ROOT}/metrics/adult-gcn-${set_type}-${suffix}-epochs200.fairness_models.csv" \
      --fairness-metric "${metric}" \
      --output-prefix "${OUTPUT_ROOT}/figures/adult-gcn-${set_type}-${suffix}-epochs200.fairness_${metric}.pareto"
  done
done

python -u -m gnn_rashomon.cli.build_report \
  --set-pattern 'adult-gcn-*-relative*-epochs200' \
  --output-dir "${OUTPUT_ROOT}" \
  --report-name adult_corrected_gcn
python -u -m gnn_rashomon.cli.plot_report \
  --output-dir "${OUTPUT_ROOT}" \
  --report-name adult_corrected_gcn

echo "Corrected Adult artifacts: ${OUTPUT_ROOT}"
