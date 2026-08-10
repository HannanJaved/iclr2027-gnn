#!/usr/bin/env bash
#SBATCH --job-name=gnn_cp_hrew_analysis
#SBATCH --time=01:00:00
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
TARGET_IDS=(0p7 0p6)
TARGET_VALUES=(0.70 0.60)

for dataset in "${DATASETS[@]}"; do
  graph_ids=()
  metric_ids=()
  delta_ids=()

  for target_id in "${TARGET_IDS[@]}"; do
    graph_path="${PROJECT_DIR}/outputs/graphs/${dataset}-homophily_targeted-target${target_id}-0p05-seed0.pt"
    set_path="outputs/rashomon_sets/${dataset}-gcn-homophily_targeted_${target_id}-relative0.1-epochs200.json"
    mult_table="outputs/metrics/${dataset}-gcn-homophily_targeted_${target_id}-relative0.1-epochs200.multiplicity_nodes.csv"

    test -f "${graph_path}"

    python -u -m gnn_rashomon.cli.generate_rashomon \
      --dataset "${dataset}" \
      --architecture gcn \
      --set-type "homophily_targeted_${target_id}" \
      --rashomon-type "homophily_targeted_${target_id}" \
      --max-epochs 200 \
      --tolerance-mode relative \
      --epsilon 0.10

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
      --output "outputs/metrics/${dataset}-homophily-targeted-${target_id}-seed0.delta_multiplicity.csv"

    graph_ids+=("homophily_targeted-target${target_id}-0p05-seed0")
    metric_ids+=("homophily_targeted_${target_id}")
    delta_ids+=("homophily-targeted-${target_id}-seed0")
  done

  python -u -m gnn_rashomon.cli.build_rewiring_report \
    --dataset "${dataset}" \
    --mode homophily_targeted \
    --strength-ids target0p7 target0p6 \
    --strength-values "${TARGET_VALUES[@]}" \
    --graph-ids "${graph_ids[@]}" \
    --metric-ids "${metric_ids[@]}" \
    --delta-ids "${delta_ids[@]}" \
    --plot-prefix "${dataset}_homophily_targeted_rewiring" \
    --x-label "Target homophily"
done
