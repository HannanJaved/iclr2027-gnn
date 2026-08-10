#!/usr/bin/env bash
#SBATCH --job-name=gnn_cora_rw_analysis
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -p alpha
#SBATCH --requeue
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1

module load CUDA
# Optional: source your local cache/env setup here
PROJECT_DIR=${PROJECT_DIR}
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv/bin/activate"

STRENGTH_IDS=(0p01 0p1 0p25)

for strength_id in "${STRENGTH_IDS[@]}"; do
  GRAPH_PATH="${PROJECT_DIR}/outputs/graphs/cora-random-${strength_id}-seed0.pt"
  SET_PATH="outputs/rashomon_sets/cora-gcn-rewired_random_${strength_id}-relative0.1-epochs200.json"
  MULT_TABLE="outputs/metrics/cora-gcn-rewired_random_${strength_id}-relative0.1-epochs200.multiplicity_nodes.csv"

  test -f "${GRAPH_PATH}"

  python -u -m gnn_rashomon.cli.generate_rashomon \
    --dataset cora \
    --architecture gcn \
    --set-type "rewired_random_${strength_id}" \
    --rashomon-type "rewired_random_${strength_id}" \
    --max-epochs 200 \
    --tolerance-mode relative \
    --epsilon 0.10

  python -u -m gnn_rashomon.cli.compute_multiplicity "${SET_PATH}"

  python -u -m gnn_rashomon.cli.compute_structure \
    --dataset cora \
    --rashomon-set "${SET_PATH}" \
    --multiplicity-table "${MULT_TABLE}" \
    --graph-path "${GRAPH_PATH}"

  python -u -m gnn_rashomon.cli.compare_rewired \
    --original-table outputs/metrics/cora-gcn-seed-relative0.1-epochs200.multiplicity_nodes.csv \
    --rewired-table "${MULT_TABLE}" \
    --output "outputs/metrics/cora-random-${strength_id}-seed0.delta_multiplicity.csv"
done
