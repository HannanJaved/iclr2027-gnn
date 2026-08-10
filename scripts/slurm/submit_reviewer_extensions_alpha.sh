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

setup_job=$(submit scripts/slurm/install_ogbn_arxiv_alpha.sbatch)
validate_job=$(submit --dependency="afterok:${setup_job}" \
  scripts/slurm/validate_ogbn_arxiv.sbatch)
seed_job=$(submit --dependency="afterok:${validate_job}" \
  scripts/slurm/seed_ogbn_arxiv_array.sbatch)
analysis_job=$(submit --dependency="afterok:${seed_job}" \
  scripts/slurm/analysis_ogbn_arxiv_stage1.sbatch)

printf '%s\n' \
  "Submitted self-contained Alpha OGB chain:" \
  "  setup:       ${setup_job}" \
  "  validation:  ${validate_job}" \
  "  seed array:  ${seed_job}" \
  "  analysis:    ${analysis_job}"
