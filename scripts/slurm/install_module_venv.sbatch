#!/usr/bin/env bash
#SBATCH --job-name=gnn_install_module
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=12G
#SBATCH -p romeo
#SBATCH --requeue
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

module purge
module load release/24.10 GCC/13.2.0 OpenMPI/4.1.6 PyTorch/2.3.0

# Optional: source your local cache/env setup here
PROJECT_DIR=${PROJECT_DIR}
cd "${PROJECT_DIR}"

python -m venv --system-site-packages "${PROJECT_DIR}/.venv-module"
source "${PROJECT_DIR}/.venv-module/bin/activate"

python -m pip install --upgrade pip setuptools wheel

# Install this project without following pyproject dependencies, because the
# system PyTorch module is the known-good torch for this cluster image.
python -m pip install --no-deps -e .
python -m pip install \
  "torch-geometric>=2.5" \
  "scikit-learn>=1.3" \
  "statsmodels>=0.14" \
  "matplotlib>=3.8" \
  "omegaconf>=2.3"

python -u - <<'PY'
import importlib
import torch

print(f"python import ok", flush=True)
print(f"torch={torch.__version__} file={torch.__file__} cuda={torch.cuda.is_available()}", flush=True)
for name in ["torch_geometric", "sklearn", "statsmodels", "matplotlib", "omegaconf"]:
    module = importlib.import_module(name)
    print(f"{name}={getattr(module, '__version__', '<no version>')} file={getattr(module, '__file__', '<namespace>')}", flush=True)
PY

python -m pytest tests/test_checkpointing.py tests/test_community_boundaries.py -q
