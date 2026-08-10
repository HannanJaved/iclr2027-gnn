#!/usr/bin/env bash
#SBATCH --job-name=gnn_cp_rw_rel2
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH -p romeo
#SBATCH --requeue
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH=${PROJECT_DIR}/src:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES=""

module purge
module load release/24.10 GCC/13.2.0 OpenMPI/4.1.6 PyTorch/2.3.0

# Optional: source your local cache/env setup here
PROJECT_DIR=${PROJECT_DIR}
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv-module/bin/activate"

export MPLCONFIGDIR="${TMPDIR:-/tmp}/gnn-rashomon-matplotlib-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/gnn-rashomon-cache-${SLURM_JOB_ID}"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

DATASETS=(citeseer pubmed)
STRENGTH_IDS=(0p01 0p05 0p1 0p25)
STRENGTH_VALUES=(0.01 0.05 0.10 0.25)
EPSILON=2.0
SET_SUFFIX=relative2-epochs200

for dataset in "${DATASETS[@]}"; do
  graph_ids=()
  metric_ids=()
  delta_ids=()

  for strength_id in "${STRENGTH_IDS[@]}"; do
    graph_path="${PROJECT_DIR}/outputs/graphs/${dataset}-random-${strength_id}-seed0.pt"
    set_path="outputs/rashomon_sets/${dataset}-gcn-rewired_random_${strength_id}-${SET_SUFFIX}.json"
    mult_table="outputs/metrics/${dataset}-gcn-rewired_random_${strength_id}-${SET_SUFFIX}.multiplicity_nodes.csv"
    delta_id="random-${strength_id}-seed0-rel2"

    test -f "${graph_path}"

    python -u -m gnn_rashomon.cli.generate_rashomon \
      --dataset "${dataset}" \
      --architecture gcn \
      --set-type "rewired_random_${strength_id}" \
      --rashomon-type "rewired_random_${strength_id}" \
      --max-epochs 200 \
      --tolerance-mode relative \
      --epsilon "${EPSILON}"

    python -u -m gnn_rashomon.cli.compute_multiplicity "${set_path}"

    python -u -m gnn_rashomon.cli.compute_structure \
      --dataset "${dataset}" \
      --rashomon-set "${set_path}" \
      --multiplicity-table "${mult_table}" \
      --graph-path "${graph_path}" \
      --include-community

    python -u -m gnn_rashomon.cli.compare_rewired \
      --original-table "outputs/metrics/${dataset}-gcn-seed-relative0.1-epochs200.multiplicity_nodes.csv" \
      --rewired-table "${mult_table}" \
      --output "outputs/metrics/${dataset}-${delta_id}.delta_multiplicity.csv"

    graph_ids+=("random-${strength_id}-seed0")
    metric_ids+=("rewired_random_${strength_id}")
    delta_ids+=("${delta_id}")
  done

  python -u -m gnn_rashomon.cli.build_rewiring_report \
    --dataset "${dataset}" \
    --mode random \
    --strength-ids "${STRENGTH_IDS[@]}" \
    --strength-values "${STRENGTH_VALUES[@]}" \
    --graph-ids "${graph_ids[@]}" \
    --metric-ids "${metric_ids[@]}" \
    --delta-ids "${delta_ids[@]}" \
    --set-suffix "${SET_SUFFIX}" \
    --plot-prefix "${dataset}_random_rewiring_rel2"
done
