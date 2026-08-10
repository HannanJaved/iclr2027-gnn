#!/usr/bin/env bash
#SBATCH --job-name=gnn_next_analysis
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

echo "Building seed Rashomon structural report"
python -u -m gnn_rashomon.cli.build_report \
  --set-pattern '*-gcn-seed-relative0.1-epochs200' \
  --report-name seed_relative0.1

echo "Building relative-2.0 hyperparameter Rashomon analyses"
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

echo "Next analysis batch complete"
