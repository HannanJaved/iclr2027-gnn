#!/usr/bin/env bash
#SBATCH --job-name=gnn_diag_torch_cpu
#SBATCH --time=00:20:00
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
export PYTHONPATH=${PROJECT_DIR}/src:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES=""

module purge
module load release/24.10 GCC/13.2.0 OpenMPI/4.1.6 PyTorch/2.3.0

# Optional: source your local cache/env setup here
PROJECT_DIR=${PROJECT_DIR}
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv-module/bin/activate"

export MPLCONFIGDIR="${TMPDIR:-/tmp}/gnn-rashomon-matplotlib-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/gnn-rashomon-cache-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

python -u - <<'PY'
from pathlib import Path
import time

def mark(label):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {label}", flush=True)

mark("cpu diagnostic start")
mark("before import torch")
import torch
mark(f"after import torch version={torch.__version__} cuda_available={torch.cuda.is_available()}")

path = Path("data/Planetoid/Cora/processed/data.pt")
mark(f"before torch.load {path}")
payload = torch.load(path, map_location="cpu", weights_only=False)
mark(f"after torch.load type={type(payload)}")

mark("before load_graph(cora)")
from gnn_rashomon.data.loaders import load_graph
graph = load_graph("cora", root="data/Planetoid")
mark(
    "after load_graph "
    f"nodes={graph.metadata.num_nodes} edges={graph.metadata.num_edges} "
    f"features={graph.metadata.num_features} classes={graph.metadata.num_classes}"
)
PY
