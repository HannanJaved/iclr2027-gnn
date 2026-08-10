#!/usr/bin/env bash
#SBATCH --job-name=gnn_canonical_eq
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -p romeo
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err

set -euo pipefail

: "${REFERENCE_ROOT:?Export REFERENCE_ROOT as the archived focal-artifact tree}"
: "${CANDIDATE_ROOT:?Export CANDIDATE_ROOT as the independently regenerated tree}"

PROJECT_DIR=${PROJECT_DIR}
REPORT_PATH="${REPORT_PATH:-${PROJECT_DIR}/outputs/metadata/canonical_environment_equivalence.json}"

[[ -d "${REFERENCE_ROOT}" ]] || { echo "Missing REFERENCE_ROOT: ${REFERENCE_ROOT}" >&2; exit 2; }
[[ -d "${CANDIDATE_ROOT}" ]] || { echo "Missing CANDIDATE_ROOT: ${CANDIDATE_ROOT}" >&2; exit 2; }
[[ "$(readlink -f "${REFERENCE_ROOT}")" != "$(readlink -f "${CANDIDATE_ROOT}")" ]] || {
  echo "Reference and candidate roots must differ" >&2
  exit 2
}

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES=""

module purge
module load release/24.10 GCC/13.2.0 OpenMPI/4.1.6 PyTorch/2.3.0
# Optional: source your local cache/env setup here
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv/bin/activate"

python -u -m gnn_rashomon.cli.compare_artifact_environments \
  --reference-root "${REFERENCE_ROOT}" \
  --candidate-root "${CANDIDATE_ROOT}" \
  --patterns '**/*.csv' '**/*.json' '**/*.pt' \
  --rtol 1e-6 \
  --atol 1e-8 \
  --output "${REPORT_PATH}"
