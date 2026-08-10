#!/usr/bin/env bash
#SBATCH --job-name=gnn_ext_final_snapshot
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
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

python -u -m gnn_rashomon.cli.build_repro_manifest \
  --output outputs/metadata/reviewer_extensions_final_manifest.json \
  --artifact "outputs/configs/ogbn_arxiv-*.json" \
  --artifact "outputs/checkpoints/ogbn_arxiv-*" \
  --artifact "outputs/predictions/ogbn_arxiv-*" \
  --artifact "outputs/metrics/ogbn_arxiv-*" \
  --artifact "outputs/rashomon_sets/ogbn_arxiv-*" \
  --artifact "outputs/figures/ogbn_arxiv*" \
  --artifact "outputs/configs/nba-*.json" \
  --artifact "outputs/checkpoints/nba-gat-*" \
  --artifact "outputs/predictions/nba-gat-*" \
  --artifact "outputs/metrics/nba-gat-*" \
  --artifact "outputs/rashomon_sets/nba-gat-*" \
  --artifact "outputs/figures/nba-gat-*" \
  --artifact "outputs/baselines/fairsin/**/*" \
  --artifact "outputs/figures/nba_fairsin_*" \
  --log-glob ".logs/gnn_ogbn_*" \
  --log-glob ".logs/gnn_nba_gat_*" \
  --log-glob ".logs/gnn_fairsin_nba_*"
