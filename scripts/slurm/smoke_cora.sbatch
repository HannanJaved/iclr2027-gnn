#!/usr/bin/env bash
#SBATCH --job-name=gnn_cora_smoke
#SBATCH --time=00:10:00
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

module load CUDA
# Optional: source your local cache/env setup here
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv/bin/activate"

date
echo "Host: $(hostname)"
echo "Python: $(python --version)"
python - <<'PY'
import torch
print(f"torch={torch.__version__}", flush=True)
print(f"cuda_available={torch.cuda.is_available()}", flush=True)
if torch.cuda.is_available():
    print(f"cuda_device={torch.cuda.get_device_name(0)}", flush=True)
PY

python - <<'PY'
import numpy as np

from gnn_rashomon.multiplicity.predictive_entropy import predictive_entropy
from gnn_rashomon.multiplicity.rashomon_capacity import rashomon_capacity
from gnn_rashomon.multiplicity.variation_ratio import variation_ratio
from gnn_rashomon.rashomon.tolerance import is_retained

p = np.array([[[1.0, 0.0], [0.5, 0.5]], [[1.0, 0.0], [0.5, 0.5]]])
assert np.allclose(predictive_entropy(p), [0.0, np.log(2.0)])
assert np.allclose(variation_ratio(p), [0.0, 0.0])
assert np.allclose(rashomon_capacity(p), [0.0, 0.0])
assert is_retained(1.0, 1.0, "absolute", 0.0)
print("fast health checks passed", flush=True)
PY

python -u -m gnn_rashomon.cli.train_baseline \
  dataset=cora \
  trainer.device=cuda \
  trainer.max_epochs=5
