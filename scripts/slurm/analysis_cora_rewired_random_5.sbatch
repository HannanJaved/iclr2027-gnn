#!/usr/bin/env bash
#SBATCH --job-name=gnn_cora_rw5_analysis
#SBATCH --time=00:15:00
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
GRAPH_PATH="${PROJECT_DIR}/outputs/graphs/cora-random-0p05-seed0.pt"
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv/bin/activate"

python -u -m gnn_rashomon.cli.generate_rashomon \
  --dataset cora \
  --architecture gcn \
  --set-type rewired_random_0p05 \
  --rashomon-type rewired_random_0p05 \
  --max-epochs 200 \
  --tolerance-mode relative \
  --epsilon 0.10

python -u -m gnn_rashomon.cli.compute_multiplicity \
  outputs/rashomon_sets/cora-gcn-rewired_random_0p05-relative0.1-epochs200.json

python -u -m gnn_rashomon.cli.compute_structure \
  --dataset cora \
  --rashomon-set outputs/rashomon_sets/cora-gcn-rewired_random_0p05-relative0.1-epochs200.json \
  --multiplicity-table outputs/metrics/cora-gcn-rewired_random_0p05-relative0.1-epochs200.multiplicity_nodes.csv \
  --graph-path "${GRAPH_PATH}"

python -u -m gnn_rashomon.cli.compare_rewired \
  --original-table outputs/metrics/cora-gcn-seed-relative0.1-epochs200.multiplicity_nodes.csv \
  --rewired-table outputs/metrics/cora-gcn-rewired_random_0p05-relative0.1-epochs200.multiplicity_nodes.csv \
  --output outputs/metrics/cora-random-0p05-seed0.delta_multiplicity.csv
