#!/usr/bin/env bash
#SBATCH --job-name=gnn_ogbn_setup
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH -p alpha
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES=""

module purge
module load release/24.10 GCC/13.2.0 OpenMPI/4.1.6 PyTorch/2.3.0
# Optional: source your local cache/env setup here
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ALPHA_DEPS="${PROJECT_DIR}/.alpha-deps"
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv/bin/activate"
mkdir -p .logs "${ALPHA_DEPS}"

# Keep the Alpha-only OGB dependency isolated from the canonical shared environment.
python -m pip install --no-deps --upgrade --target "${ALPHA_DEPS}" "ogb>=1.3.6"
PYTHONPATH="${PROJECT_DIR}/src:${ALPHA_DEPS}:${PYTHONPATH:-}" python - <<'PY'
import ogb
import torch

print(f"ogb={ogb.__version__}", flush=True)
print(f"torch={torch.__version__}", flush=True)
print(f"cuda_available={torch.cuda.is_available()}", flush=True)
PY
PYTHONPATH="${PROJECT_DIR}/src:${ALPHA_DEPS}:${PYTHONPATH:-}" \
  python -m pytest -q tests/test_ogbn_arxiv_loader.py
