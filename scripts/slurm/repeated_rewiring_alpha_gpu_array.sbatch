#!/usr/bin/env bash
#SBATCH --job-name=gnn_repeat_rw
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH -p alpha
#SBATCH --requeue
#SBATCH --array=0-20%3
#SBATCH --output=.logs/%x_%A_%a.out
#SBATCH --error=.logs/%x_%A_%a.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

module purge
module load release/24.10 GCC/13.2.0 OpenMPI/4.1.6 PyTorch/2.3.0
# Optional: source your local cache/env setup here
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv-module/bin/activate"

DATASETS=(cora citeseer pubmed)
task="${SLURM_ARRAY_TASK_ID}"

if (( task < 3 )); then
  dataset="${DATASETS[$task]}"
  mode=local_entanglement
  output_root=outputs/repeated_rewiring/local
elif (( task < 15 )); then
  index=$((task - 3))
  dataset="${DATASETS[$((index / 4))]}"
  STRENGTHS=(0.01 0.05 0.10 0.25)
  STRENGTH_IDS=(0p01 0p05 0p1 0p25)
  strength="${STRENGTHS[$((index % 4))]}"
  strength_id="${STRENGTH_IDS[$((index % 4))]}"
  mode=random
  output_root=outputs/repeated_rewiring/random
else
  index=$((task - 15))
  dataset="${DATASETS[$((index / 2))]}"
  TARGETS=(0.72 0.62)
  TARGET_IDS=(0p72 0p62)
  target="${TARGETS[$((index % 2))]}"
  target_id="${TARGET_IDS[$((index % 2))]}"
  mode=homophily_targeted
  output_root=outputs/repeated_rewiring/homophily
fi

for seed in $(seq 0 19); do
  if [[ "${mode}" == local_entanglement ]]; then
    graph_id="${dataset}-local_entanglement-tau0p1-0p05-seed${seed}"
    python -u -m gnn_rashomon.cli.run_rewiring \
      --dataset "${dataset}" --mode "${mode}" --strength 0.05 \
      --tau 0.10 --delta-e 0.02 --max-feature-drift 0.02 \
      --treatment-count 50 --min-degree 2 --max-attempts 50000 \
      --seed "${seed}" --output-dir "${output_root}"
  elif [[ "${mode}" == random ]]; then
    graph_id="${dataset}-random-${strength_id}-seed${seed}"
    python -u -m gnn_rashomon.cli.run_rewiring \
      --dataset "${dataset}" --mode "${mode}" --strength "${strength}" \
      --seed "${seed}" --output-dir "${output_root}"
  else
    graph_id="${dataset}-homophily_targeted-target${target_id}-0p05-seed${seed}"
    python -u -m gnn_rashomon.cli.run_rewiring \
      --dataset "${dataset}" --mode "${mode}" --target-homophily "${target}" \
      --tolerance 0.02 --max-attempts 75000 \
      --seed "${seed}" --output-dir "${output_root}"
  fi

  python -u -m gnn_rashomon.cli.evaluate_fixed_set_rewiring \
    --rashomon-set "outputs/rashomon_sets/${dataset}-gcn-seed-relative0.1-epochs200.json" \
    --original-node-table "outputs/metrics/${dataset}-gcn-seed-relative0.1-epochs200.multiplicity_nodes.csv" \
    --graph-path "${output_root}/graphs/${graph_id}.pt" \
    --rewiring-metadata "${output_root}/graphs/${graph_id}.json" \
    --output "${output_root}/realizations/${graph_id}.json" \
    --device cuda
done
