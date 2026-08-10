#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR}
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/release/adult_corrected_$(date +%Y%m%d_%H%M%S)}"
cd "${PROJECT_DIR}"

if [[ -e "${OUTPUT_ROOT}" ]]; then
  echo "Refusing to mix corrected runs into an existing output root: ${OUTPUT_ROOT}" >&2
  echo "Choose a fresh OUTPUT_ROOT or omit it to use a timestamped path." >&2
  exit 2
fi

prep_job=$(sbatch --parsable \
  --export="ALL,OUTPUT_ROOT=${OUTPUT_ROOT}" \
  scripts/slurm/adult_corrected_prepare_romeo_cpu.sbatch)
seed_job=$(sbatch --parsable \
  --dependency="afterok:${prep_job}" \
  --export="ALL,OUTPUT_ROOT=${OUTPUT_ROOT}" \
  scripts/slurm/adult_corrected_seed_romeo_cpu_array.sbatch)
hp_job=$(sbatch --parsable \
  --dependency="afterok:${prep_job}" \
  --export="ALL,OUTPUT_ROOT=${OUTPUT_ROOT}" \
  scripts/slurm/adult_corrected_hyperparameter_romeo_cpu_array.sbatch)
analysis_job=$(sbatch --parsable \
  --dependency="afterok:${seed_job}:${hp_job}" \
  --export="ALL,OUTPUT_ROOT=${OUTPUT_ROOT}" \
  scripts/slurm/adult_corrected_analysis_romeo_cpu.sbatch)
inference_job=$(sbatch --parsable \
  --dependency="afterok:${analysis_job}" \
  --export="ALL,OUTPUT_ROOT=${OUTPUT_ROOT}" \
  scripts/slurm/adult_corrected_fairness_inference_romeo_cpu.sbatch)

printf 'Adult prepare:  %s\n' "${prep_job}"
printf 'Adult seeds:    %s\n' "${seed_job}"
printf 'Adult hyper:    %s\n' "${hp_job}"
printf 'Adult analysis: %s\n' "${analysis_job}"
printf 'Adult inference: %s\n' "${inference_job}"
printf 'Output root:    %s\n' "${OUTPUT_ROOT}"
