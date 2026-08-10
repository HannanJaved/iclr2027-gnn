#!/usr/bin/env bash
#SBATCH --job-name=gnn_cp_lent_analysis
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

for dataset in "${DATASETS[@]}"; do
  graph_path="${PROJECT_DIR}/outputs/graphs/${dataset}-local_entanglement-tau0p1-0p05-seed0.pt"
  set_path="outputs/rashomon_sets/${dataset}-gcn-local_entanglement_tau0p1-relative2-epochs200.json"
  mult_table="outputs/metrics/${dataset}-gcn-local_entanglement_tau0p1-relative2-epochs200.multiplicity_nodes.csv"
  delta_table="outputs/metrics/${dataset}-local-entanglement-tau0p1-relative2-seed0.delta_multiplicity.csv"

  test -f "${graph_path}"

  python -u -m gnn_rashomon.cli.generate_rashomon \
    --dataset "${dataset}" \
    --architecture gcn \
    --set-type "local_entanglement_tau0p1" \
    --rashomon-type "local_entanglement_tau0p1" \
    --max-epochs 200 \
    --tolerance-mode relative \
    --epsilon 2.0

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
    --output "${delta_table}"

  python -u -m gnn_rashomon.cli.build_rewiring_report \
    --dataset "${dataset}" \
    --mode local_entanglement \
    --strength-ids tau0p1 \
    --strength-values 0.10 \
    --graph-ids local_entanglement-tau0p1-0p05-seed0 \
    --metric-ids local_entanglement_tau0p1 \
    --delta-ids local-entanglement-tau0p1-relative2-seed0 \
    --set-suffix relative2-epochs200 \
    --plot-prefix "${dataset}_local_entanglement_rewiring" \
    --x-label "Treatment entropy target tau"

  python -u -m gnn_rashomon.cli.analyze_local_entanglement \
    --rewiring-metadata "outputs/graphs/${dataset}-local_entanglement-tau0p1-0p05-seed0.json" \
    --original-structure "outputs/metrics/${dataset}-gcn-seed-relative0.1-epochs200.structure_nodes.csv" \
    --rewired-structure "outputs/metrics/${dataset}-gcn-local_entanglement_tau0p1-relative2-epochs200.structure_nodes.csv" \
    --delta-table "${delta_table}" \
    --output-prefix "outputs/figures/${dataset}_local_entanglement_treatment_control"
done
