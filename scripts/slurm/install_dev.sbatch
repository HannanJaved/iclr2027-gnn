#!/usr/bin/env bash
#SBATCH --job-name=gnn_install
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH -p alpha
#SBATCH --requeue
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err

set -euo pipefail

module load CUDA
# Optional: source your local cache/env setup here
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv/bin/activate"

python -m pip install --upgrade pip
python - <<'PY'
import subprocess
import sys

prefixes = ("torch==", "triton==", "cuda-", "nvidia-")
freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
packages = []
for line in freeze.splitlines():
    lowered = line.lower()
    if lowered.startswith(prefixes):
        packages.append(line.split("==", 1)[0])

if packages:
    print(f"Removing incompatible torch/CUDA packages: {packages}", flush=True)
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", *packages])
PY
python -m pip install -e ".[dev]"
python - <<'PY'
import torch

print(f"torch={torch.__version__}", flush=True)
print(f"cuda_available={torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"cuda_device={torch.cuda.get_device_name(0)}", flush=True)
PY
python -m pytest -q
