#!/usr/bin/env bash
#SBATCH --job-name=gnn_ogbn_analysis
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH -p alpha
#SBATCH --requeue
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH=${PROJECT_DIR}/src:${PROJECT_DIR}/.alpha-deps:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
module purge
module load release/24.10 GCC/13.2.0 OpenMPI/4.1.6 PyTorch/2.3.0
# Optional: source your local cache/env setup here
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv/bin/activate"
mkdir -p .logs

python -u -m gnn_rashomon.cli.audit_rashomon_health \
  --dataset ogbn_arxiv \
  --architecture gcn \
  --rashomon-type seed_ogbn_revised \
  --max-epochs 300 \
  --expected-candidates 50 \
  --tolerance-mode relative \
  --epsilon 0.10 \
  --output-prefix outputs/figures/ogbn_arxiv_revised_seed_health

python -u -m gnn_rashomon.cli.generate_rashomon \
  --dataset ogbn_arxiv \
  --architecture gcn \
  --set-type seed \
  --rashomon-type seed_ogbn_revised \
  --max-epochs 300 \
  --tolerance-mode relative \
  --epsilon 0.10

SET_ID=ogbn_arxiv-gcn-seed-relative0.1-epochs300
python -u -m gnn_rashomon.cli.compute_multiplicity \
  "outputs/rashomon_sets/${SET_ID}.json" \
  --capacity-method chunked \
  --capacity-chunk-size 4
python -u -m gnn_rashomon.cli.compute_structure \
  --dataset ogbn_arxiv \
  --rashomon-set "outputs/rashomon_sets/${SET_ID}.json" \
  --multiplicity-table "outputs/metrics/${SET_ID}.multiplicity_nodes.csv" \
  --profile scalable
python -u -m gnn_rashomon.cli.build_report \
  --set-pattern "${SET_ID}" \
  --report-name ogbn_arxiv_stage1
