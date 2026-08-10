#!/usr/bin/env bash
#SBATCH --job-name=gat_rw_prepare
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH -p romeo
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES=""

module purge
module load release/24.10 GCC/13.2.0 OpenMPI/4.1.6 PyTorch/2.3.0
# Optional: source your local cache/env setup here
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv-module/bin/activate"

python -u -m gnn_rashomon.cli.generate_rashomon \
  --dataset pubmed \
  --architecture gat \
  --set-type canonical_seed \
  --rashomon-type canonical_arch_gat \
  --output-dir outputs/canonical_pubmed_gat \
  --max-epochs 200 \
  --tolerance-mode relative \
  --epsilon 2.0

python -u -m gnn_rashomon.cli.build_performance_sensitivity \
  --rashomon-set outputs/canonical_pubmed_gat/rashomon_sets/pubmed-gat-canonical_seed-relative2-epochs200.json \
  --validation-accuracy-delta 0.02 \
  --output-dir outputs/canonical_pubmed_gat/performance_sensitivity

python -u -m gnn_rashomon.cli.compute_multiplicity \
  outputs/canonical_pubmed_gat/performance_sensitivity/rashomon_sets/pubmed-gat-canonical_seed-relative2-epochs200-valaccdelta0p02.json \
  --output-dir outputs/non_gcn_rewiring/pubmed_gat

python -u -m gnn_rashomon.cli.analyze_set_size_robustness \
  --rashomon-set outputs/canonical_pubmed_gat/performance_sensitivity/rashomon_sets/pubmed-gat-canonical_seed-relative2-epochs200-valaccdelta0p02.json \
  --sample-size 10 \
  --draws 500 \
  --seed 20260716 \
  --output-prefix outputs/set_size_robustness/pubmed-gat-canonical_seed-relative2-epochs200-valaccdelta0p02-k10
