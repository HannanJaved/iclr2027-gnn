#!/usr/bin/env bash
#SBATCH --job-name=gnn_diag_torch
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH -p alpha
#SBATCH --requeue
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH=${PROJECT_DIR}/src:${PYTHONPATH:-}

module load CUDA
# Optional: source your local cache/env setup here
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv/bin/activate"

python -u - <<'PY'
from pathlib import Path
import time

def mark(label):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {label}", flush=True)

mark("diagnostic start")
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
