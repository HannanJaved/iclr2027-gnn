#!/usr/bin/env bash
#SBATCH --job-name=gnn_validate_data
#SBATCH --time=00:30:00
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

python -u -m gnn_rashomon.cli.validate_dataset dataset=cora
python -u -m gnn_rashomon.cli.validate_dataset dataset=citeseer
python -u -m gnn_rashomon.cli.validate_dataset dataset=pubmed

if [[ -f data/Adult/adult.csv || -f data/Adult/adult.data || -f data/Adult/raw/adult.csv ]]; then
  python -u -m gnn_rashomon.cli.validate_dataset dataset=adult
else
  echo "Skipping Adult validation because no local Adult CSV/data file is present under data/Adult"
fi

echo "Dataset validation complete"
