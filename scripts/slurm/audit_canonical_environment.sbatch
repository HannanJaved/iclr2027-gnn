#!/usr/bin/env bash
#SBATCH --job-name=gnn_env_audit
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -p romeo
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err

set -euo pipefail
: "${REFERENCE_ROOT:?Set REFERENCE_ROOT to the archived analysis-artifact root}"
: "${CANDIDATE_ROOT:?Set CANDIDATE_ROOT to the canonical-environment rerun root}"
export PYTHONUNBUFFERED=1
export PYTHONPATH=${PROJECT_DIR}/src:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=""
module purge
module load release/24.10 GCC/13.2.0 OpenMPI/4.1.6 PyTorch/2.3.0
# Optional: source your local cache/env setup here
PROJECT_DIR=${PROJECT_DIR}
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv/bin/activate"
python -u -m gnn_rashomon.cli.compare_artifact_environments \
  --reference-root "${REFERENCE_ROOT}" \
  --candidate-root "${CANDIDATE_ROOT}" \
  --rtol 1e-6 --atol 1e-8 \
  --output outputs/metadata/canonical_environment_equivalence.json
