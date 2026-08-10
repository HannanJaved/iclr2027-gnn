#!/usr/bin/env bash
#SBATCH --job-name=gat_rw_summary
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
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

python -u -m gnn_rashomon.cli.summarize_repeated_rewiring \
  --realization-glob 'outputs/non_gcn_rewiring/pubmed_gat/realizations/*.json' \
  --output-prefix outputs/non_gcn_rewiring/pubmed_gat/pubmed_gat_repeated_rewiring \
  --bootstrap-samples 10000 \
  --seed 20260716

python -u -m gnn_rashomon.cli.summarize_set_size_robustness \
  --summary-glob 'outputs/set_size_robustness/*.summary.json' \
  --output outputs/set_size_robustness/set_size_robustness_summary.csv
