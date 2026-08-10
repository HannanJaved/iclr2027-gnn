#!/usr/bin/env bash
#SBATCH --job-name=gnn_smoke_summary
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --gres=gpu:1
#SBATCH -p alpha
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH=${PROJECT_DIR}/src:${PYTHONPATH:-}
module purge
module load release/24.10 GCC/13.2.0 OpenMPI/4.1.6 PyTorch/2.3.0
# Optional: source your local cache/env setup here
PROJECT_DIR=${PROJECT_DIR}
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv/bin/activate"

python -u -m gnn_rashomon.cli.audit_rashomon_health \
  --dataset ogbn_arxiv --architecture gcn --rashomon-type diagnostic_ogbn --max-epochs 300 \
  --expected-candidates 6 \
  --tolerance-mode relative --epsilon 0.05 0.1 0.2 \
  --output-prefix outputs/figures/ogbn_arxiv_recipe_smoke_health

python -u -m gnn_rashomon.cli.audit_rashomon_health \
  --dataset nba --architecture gat --rashomon-type diagnostic_nba_gat --max-epochs 300 \
  --expected-candidates 12 \
  --tolerance-mode relative --epsilon 0.05 0.1 0.2 0.5 \
  --output-prefix outputs/figures/nba_gat_recipe_smoke_health
