#!/usr/bin/env bash
#SBATCH --job-name=gnn_citeseer_hp_rset
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

python -u -m gnn_rashomon.cli.generate_rashomon \
  --dataset citeseer \
  --architecture gcn \
  --set-type hyperparameter \
  --rashomon-type hyperparameter \
  --max-epochs 200 \
  --tolerance-mode relative \
  --epsilon 0.10 \
  --no-deduplicate-seeds

python -u -m gnn_rashomon.cli.compute_multiplicity \
  outputs/rashomon_sets/citeseer-gcn-hyperparameter-relative0.1-epochs200.json
