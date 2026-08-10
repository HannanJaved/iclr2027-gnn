#!/usr/bin/env bash
#SBATCH --job-name=gnn_hp_rel2
#SBATCH --time=00:20:00
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
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv/bin/activate"

for dataset in cora pubmed citeseer; do
  python -u -m gnn_rashomon.cli.generate_rashomon \
    --dataset "${dataset}" \
    --architecture gcn \
    --set-type hyperparameter \
    --rashomon-type hyperparameter \
    --max-epochs 200 \
    --tolerance-mode relative \
    --epsilon 2.0 \
    --no-deduplicate-seeds

  python -u -m gnn_rashomon.cli.compute_multiplicity \
    "outputs/rashomon_sets/${dataset}-gcn-hyperparameter-relative2-epochs200.json"

  python -u -m gnn_rashomon.cli.compute_structure \
    --dataset "${dataset}" \
    --rashomon-set "outputs/rashomon_sets/${dataset}-gcn-hyperparameter-relative2-epochs200.json" \
    --multiplicity-table "outputs/metrics/${dataset}-gcn-hyperparameter-relative2-epochs200.multiplicity_nodes.csv"
done

python -u -m gnn_rashomon.cli.build_report \
  --set-pattern '*-gcn-hyperparameter-relative2-epochs200' \
  --report-name hyperparameter_relative2
