#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_DIR}"
mkdir -p .logs

submit() {
  local output
  if ! output=$(sbatch --parsable "$@"); then
    printf 'Submission failed: sbatch %s\n' "$*" >&2
    return 1
  fi
  if [[ -z "${output%%;*}" ]]; then
    printf 'Submission failed: sbatch returned no job ID for %s\n' "$*" >&2
    return 1
  fi
  printf '%s\n' "${output%%;*}"
}

nba_gat_seed_job=$(submit scripts/slurm/seed_nba_gat_array.sbatch)
nba_gat_hp_job=$(submit scripts/slurm/hyperparameter_nba_gat_array.sbatch)
nba_gat_analysis_job=$(submit \
  --dependency="afterok:${nba_gat_seed_job}:${nba_gat_hp_job}" \
  scripts/slurm/analysis_nba_gat_fairness.sbatch)
fairsin_job=$(submit scripts/slurm/fairsin_nba_baseline_alpha.sbatch)

printf '%s\n' \
  "Submitted self-contained Alpha NBA fairness chain:" \
  "  NBA GAT seed array: ${nba_gat_seed_job}" \
  "  NBA GAT hp array:   ${nba_gat_hp_job}" \
  "  NBA GAT analysis:   ${nba_gat_analysis_job}" \
  "  NBA FairSIN:        ${fairsin_job}"
