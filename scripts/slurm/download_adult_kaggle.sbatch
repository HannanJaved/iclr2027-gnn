#!/usr/bin/env bash
#SBATCH --job-name=gnn_adult_download
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -p romeo
#SBATCH --requeue
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH=${PROJECT_DIR}/src:${PYTHONPATH:-}

module purge
module load release/24.10 GCC/13.2.0 OpenMPI/4.1.6 PyTorch/2.3.0

# Optional: source your local cache/env setup here
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv-module/bin/activate"

if ! python - <<'PY'
try:
    import kagglehub  # noqa: F401
except ModuleNotFoundError:
    raise SystemExit(1)
PY
then
  python -m pip install kagglehub
fi

python -u -m gnn_rashomon.cli.download_adult \
  --dataset wenruliu/adult-income-dataset \
  --output data/Adult/adult.csv
