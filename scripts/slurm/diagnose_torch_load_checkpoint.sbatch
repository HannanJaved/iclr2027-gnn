#!/usr/bin/env bash
#SBATCH --job-name=gnn_diag_torch_load
#SBATCH --time=00:10:00
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
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv-module/bin/activate"

python -u - <<'PY'
from pathlib import Path
import json
import time

def mark(label):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {label}", flush=True)

mark("diagnostic start")
mark("before import torch")
import torch
mark(f"after import torch version={torch.__version__} cuda_available={torch.cuda.is_available()}")

graph_path = Path("data/Planetoid/Cora/processed/data.pt")
mark(f"before torch.load graph {graph_path}")
graph_payload = torch.load(graph_path, map_location="cpu", weights_only=False)
mark(f"after torch.load graph type={type(graph_payload)}")

rashomon = json.loads(Path("outputs/rashomon_sets/cora-gcn-seed-relative0.1-epochs200.json").read_text())
run_id = sorted(rashomon["retained_run_ids"], key=lambda rid: rashomon["runs"][rid]["train_loss"])[0]
record_path = Path(rashomon["runs"][run_id]["record_path"])
record = json.loads(record_path.read_text())
checkpoint_path = Path(record["checkpoint_path"])
mark(f"before torch.load checkpoint run_id={run_id} path={checkpoint_path}")
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
mark(f"after torch.load checkpoint keys={sorted(checkpoint.keys())}")
mark("diagnostic complete")
PY
