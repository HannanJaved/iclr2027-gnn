#!/usr/bin/env bash
# Submit #1 counterfactual-retraining analysis on Romeo (CPU only).
# Requires the alpha GPU array to have finished writing
# outputs/counterfactual_retraining/metrics/*.json
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR}
cd "${PROJECT_DIR}"

if [[ ! -d outputs/counterfactual_retraining/metrics ]]; then
  printf 'error: outputs/counterfactual_retraining/metrics is missing.\n' >&2
  printf 'Submit and finish scripts/slurm/cf_retrain_seed_alpha_gpu_array.sbatch on alpha first.\n' >&2
  exit 1
fi

analyze_job=$(sbatch --parsable scripts/slurm/cf_retrain_analyze_romeo_cpu.sbatch)

printf 'cf-retrain analyze: %s\n' "${analyze_job}"
printf 'Builds Rashomon sets + multiplicity, then writes cf_retrain_summary_*.csv\n'
