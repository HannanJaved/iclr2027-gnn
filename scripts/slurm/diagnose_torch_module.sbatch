#!/usr/bin/env bash
#SBATCH --job-name=gnn_diag_torch_module
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
export PYTHONFAULTHANDLER=1
export PYTHONPATH=${PROJECT_DIR}/src:${PYTHONPATH:-}

module purge
module load release/24.10 GCC/13.2.0 OpenMPI/4.1.6 PyTorch/2.3.0

which python
python --version
module list

python -u - <<'PY'
import faulthandler
import importlib.util
import sys
import time

def mark(label):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {label}", flush=True)

faulthandler.enable(file=sys.stderr, all_threads=True)
faulthandler.dump_traceback_later(60, repeat=True, file=sys.stderr)

mark("before import torch")
import torch
mark(f"after import torch version={torch.__version__} file={torch.__file__} cuda={torch.cuda.is_available()}")

for name in ["torch_geometric", "numpy", "pandas", "sklearn", "scipy", "yaml", "statsmodels"]:
    spec = importlib.util.find_spec(name)
    mark(f"find_spec {name}: {None if spec is None else spec.origin}")
    if spec is not None:
        module = __import__(name)
        mark(f"imported {name} from {getattr(module, '__file__', '<namespace>')}")
PY
