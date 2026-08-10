#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR=${PROJECT_DIR}
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

ogbn_seed_job=$(submit scripts/slurm/seed_ogbn_arxiv_array.sbatch)
ogbn_analysis_job=$(submit --dependency="afterok:${ogbn_seed_job}" \
  scripts/slurm/analysis_ogbn_arxiv_stage1.sbatch)

nba_seed_job=$(submit scripts/slurm/seed_nba_gat_array.sbatch)
nba_hp_job=$(submit scripts/slurm/hyperparameter_nba_gat_array.sbatch)
nba_analysis_job=$(submit \
  --dependency="afterok:${nba_seed_job}:${nba_hp_job}" \
  scripts/slurm/analysis_nba_gat_fairness.sbatch)

snapshot_job=$(submit \
  --dependency="afterok:${ogbn_analysis_job}:${nba_analysis_job}" \
  scripts/slurm/snapshot_reviewer_extensions_final_alpha.sbatch)

printf '%s\n' \
  "Submitted smoke-selected reviewer-extension reruns:" \
  "  OGB seed array:       ${ogbn_seed_job}" \
  "  OGB analysis:         ${ogbn_analysis_job}" \
  "  NBA GAT seed array:   ${nba_seed_job}" \
  "  NBA GAT hp array:     ${nba_hp_job}" \
  "  NBA GAT analysis:     ${nba_analysis_job}" \
  "  final snapshot:       ${snapshot_job}"
