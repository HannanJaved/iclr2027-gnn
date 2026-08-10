#!/usr/bin/env bash
#SBATCH --job-name=gnn_gcn_comm_struct
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH -p alpha
#SBATCH --requeue
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH=${PROJECT_DIR}/src:${PYTHONPATH:-}

module load CUDA
# Optional: source your local cache/env setup here
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv/bin/activate"

for dataset in cora pubmed citeseer; do
  for set_type in seed hyperparameter; do
    if [[ "${set_type}" == "seed" ]]; then
      suffix="relative0.1"
    else
      suffix="relative2"
    fi
    set_path="outputs/rashomon_sets/${dataset}-gcn-${set_type}-${suffix}-epochs200.json"
    mult_path="outputs/metrics/${dataset}-gcn-${set_type}-${suffix}-epochs200.multiplicity_nodes.csv"
    test -f "${set_path}"
    test -f "${mult_path}"
    python -u -m gnn_rashomon.cli.compute_structure \
      --dataset "${dataset}" \
      --rashomon-set "${set_path}" \
      --multiplicity-table "${mult_path}" \
      --include-community
  done
done
