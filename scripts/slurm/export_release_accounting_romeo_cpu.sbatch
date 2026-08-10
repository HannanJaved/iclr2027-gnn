#!/usr/bin/env bash
#SBATCH --job-name=gnn_release_acct
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH -p romeo
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ADULT_ROOT="${ADULT_ROOT:?Export ADULT_ROOT from the corrected Adult pipeline}"
OUTPUT_PATH="${OUTPUT_PATH:-${ADULT_ROOT}/metadata/adult_corrected_slurm_accounting.tsv}"
JOB_IDS_PATH="${JOB_IDS_PATH:-${ADULT_ROOT}/metadata/adult_corrected_slurm_job_ids.txt}"

cd "${PROJECT_DIR}"
mkdir -p "$(dirname "${OUTPUT_PATH}")"

mapfile -t JOB_IDS < <(
  find .logs -maxdepth 1 -type f -name 'gnn_adult_fix_*.out' -printf '%f\n' |
    while IFS= read -r log_name; do
      if [[ "${log_name}" =~ _([0-9]+)_[0-9]+\.out$ ]]; then
        printf '%s\n' "${BASH_REMATCH[1]}"
      elif [[ "${log_name}" =~ _([0-9]+)\.out$ ]]; then
        printf '%s\n' "${BASH_REMATCH[1]}"
      fi
    done |
    sort -u
)

(( ${#JOB_IDS[@]} > 0 )) || {
  echo "No Adult job IDs found in .logs" >&2
  exit 2
}

printf '%s\n' "${JOB_IDS[@]}" > "${JOB_IDS_PATH}"
JOB_ID_LIST="$(IFS=,; echo "${JOB_IDS[*]}")"

sacct \
  --jobs="${JOB_ID_LIST}" \
  --duplicates \
  --parsable2 \
  --noheader \
  --format=JobIDRaw,JobName,Partition,State,ExitCode,Elapsed,TotalCPU,MaxRSS,AllocTRES,Start,End \
  > "${OUTPUT_PATH}"

[[ -s "${OUTPUT_PATH}" ]] || {
  echo "Accounting query returned no rows for job IDs ${JOB_ID_LIST}" >&2
  exit 2
}

echo "Queried Adult job IDs: ${JOB_ID_LIST}"
echo "Wrote $(wc -l < "${OUTPUT_PATH}") accounting rows to ${OUTPUT_PATH}"
