#!/usr/bin/env bash
# Submit #1 counterfactual retraining on alpha as ONE GPU job (all 2250 tasks).
# After it finishes, submit analysis separately on Romeo:
#   bash scripts/slurm/submit_cf_retrain_analyze_romeo.sh
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_DIR}"

train_job=$(sbatch --parsable scripts/slurm/cf_retrain_seed_alpha_gpu.sbatch)

printf 'cf-retrain monolithic GPU job: %s\n' "${train_job}"
printf 'Runs tasks 0-2249 inside one allocation (skip-if-done for resume).\n'
printf 'Optional range override example:\n'
printf '  CF_RETRAIN_START_TASK=0 CF_RETRAIN_END_TASK=749 sbatch scripts/slurm/cf_retrain_seed_alpha_gpu.sbatch\n'
printf 'When training completes, on Romeo run:\n'
printf '  bash scripts/slurm/submit_cf_retrain_analyze_romeo.sh\n'
