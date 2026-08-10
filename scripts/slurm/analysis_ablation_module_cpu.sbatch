#!/usr/bin/env bash
#SBATCH --job-name=gnn_ablate_mod_analysis
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
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

DATASETS=(cora pubmed citeseer)
ARCHITECTURES=(mlp topology_only)

for architecture in "${ARCHITECTURES[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    echo "Analyzing seed ablation dataset=${dataset} architecture=${architecture}"
    python -u -m gnn_rashomon.cli.generate_rashomon \
      --dataset "${dataset}" \
      --architecture "${architecture}" \
      --set-type seed \
      --rashomon-type seed \
      --max-epochs 200 \
      --tolerance-mode relative \
      --epsilon 0.10

    python -u -m gnn_rashomon.cli.compute_multiplicity \
      "outputs/rashomon_sets/${dataset}-${architecture}-seed-relative0.1-epochs200.json"

    python -u -m gnn_rashomon.cli.compute_structure \
      --dataset "${dataset}" \
      --rashomon-set "outputs/rashomon_sets/${dataset}-${architecture}-seed-relative0.1-epochs200.json" \
      --multiplicity-table "outputs/metrics/${dataset}-${architecture}-seed-relative0.1-epochs200.multiplicity_nodes.csv"

    echo "Analyzing hyperparameter ablation dataset=${dataset} architecture=${architecture}"
    python -u -m gnn_rashomon.cli.generate_rashomon \
      --dataset "${dataset}" \
      --architecture "${architecture}" \
      --set-type hyperparameter \
      --rashomon-type hyperparameter \
      --max-epochs 200 \
      --tolerance-mode relative \
      --epsilon 2.0 \
      --no-deduplicate-seeds

    python -u -m gnn_rashomon.cli.compute_multiplicity \
      "outputs/rashomon_sets/${dataset}-${architecture}-hyperparameter-relative2-epochs200.json"

    python -u -m gnn_rashomon.cli.compute_structure \
      --dataset "${dataset}" \
      --rashomon-set "outputs/rashomon_sets/${dataset}-${architecture}-hyperparameter-relative2-epochs200.json" \
      --multiplicity-table "outputs/metrics/${dataset}-${architecture}-hyperparameter-relative2-epochs200.multiplicity_nodes.csv"
  done

  python -u -m gnn_rashomon.cli.build_report \
    --set-pattern "*-${architecture}-seed-relative0.1-epochs200" \
    --report-name "${architecture}_seed_relative0.1"
  python -u -m gnn_rashomon.cli.plot_report \
    --report-name "${architecture}_seed_relative0.1"

  python -u -m gnn_rashomon.cli.build_report \
    --set-pattern "*-${architecture}-hyperparameter-relative2-epochs200" \
    --report-name "${architecture}_hyperparameter_relative2"
  python -u -m gnn_rashomon.cli.plot_report \
    --report-name "${architecture}_hyperparameter_relative2"
done

echo "Ablation analysis complete"
