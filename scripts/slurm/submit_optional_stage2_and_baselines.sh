#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."
mkdir -p .logs outputs/figures

echo "Submitting tasks that can run from existing scripts/artifacts."

PADJUST=$(sbatch --parsable scripts/slurm/statistical_correction_stage1.sbatch)
MIDDLE=$(sbatch --parsable scripts/slurm/causal_middle_link_stage1.sbatch)

NBA_JOBS=""
if [ -f scripts/slurm/smoke_nba_module_cpu.sbatch ] && \
   [ -f scripts/slurm/seed_nba_module_cpu_array.sbatch ] && \
   [ -f scripts/slurm/analysis_nba_gcn.sbatch ]; then
  NBA_SMOKE=$(sbatch --parsable scripts/slurm/smoke_nba_module_cpu.sbatch)
  NBA_SEEDS=$(sbatch --parsable --dependency=afterok:${NBA_SMOKE} scripts/slurm/seed_nba_module_cpu_array.sbatch)
  NBA_ANALYSIS=$(sbatch --parsable --dependency=afterok:${NBA_SEEDS} scripts/slurm/analysis_nba_gcn.sbatch)
  NBA_JOBS="${NBA_SMOKE}:${NBA_SEEDS}:${NBA_ANALYSIS}"
else
  echo "NBA full-pipeline scripts not found; not submitting NBA jobs."
fi

BASELINE_JOBS=""
for s in \
  scripts/slurm/fairgnn_baseline_adult.sbatch \
  scripts/slurm/fairgnn_baseline_german_credit.sbatch \
  scripts/slurm/baseline_fairgnn_stage1.sbatch \
  scripts/slurm/fairness_baseline_stage1.sbatch
do
  if [ -f "$s" ]; then
    JOB=$(sbatch --parsable "$s")
    BASELINE_JOBS="${BASELINE_JOBS:+${BASELINE_JOBS}:}${JOB}"
  fi
done
if [ -z "$BASELINE_JOBS" ]; then
  echo "FairGNN/baseline scripts not found; not submitting baseline jobs."
fi

echo "Multiple-testing correction: ${PADJUST}"
echo "Causal middle-link audit:    ${MIDDLE}"
echo "NBA jobs:                    ${NBA_JOBS:-none}"
echo "Baseline jobs:               ${BASELINE_JOBS:-none}"
