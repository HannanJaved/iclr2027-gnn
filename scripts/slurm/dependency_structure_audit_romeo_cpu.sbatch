#!/usr/bin/env bash
#SBATCH --job-name=gnn_dep_audit
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -p romeo
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR}
ADULT_ROOT="${ADULT_ROOT:-${PROJECT_DIR}/outputs/release/adult_corrected_20260715_124823}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES=""
export MPLCONFIGDIR="${TMPDIR:-/tmp}/gnn-dependency-audit-matplotlib-${SLURM_JOB_ID}"
export XDG_CACHE_HOME="${TMPDIR:-/tmp}/gnn-dependency-audit-cache-${SLURM_JOB_ID}"

module purge
module load release/24.10 GCC/13.2.0 OpenMPI/4.1.6 PyTorch/2.3.0
# Optional: source your local cache/env setup here
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv/bin/activate"
mkdir -p "${MPLCONFIGDIR}" "${XDG_CACHE_HOME}"

metadata_args=()
for prefix in \
  cora-gcn-seed-relative0.1-epochs200.gnnexplainer \
  cora-gat-arch_gat-relative2-epochs200.gnnexplainer \
  cora-appnp-arch_appnp-relative2-epochs200.gnnexplainer \
  pubmed-gat-arch_gat-relative2-epochs200.gnnexplainer \
  pubmed-appnp-arch_appnp-relative2-epochs200.gnnexplainer \
  cora-gcn-seed-relative0.1-epochs200.pgexplainer \
  cora-gat-arch_gat-relative2-epochs200.pgexplainer \
  cora-appnp-arch_appnp-relative2-epochs200.pgexplainer \
  pubmed-gat-arch_gat-relative2-epochs200.pgexplainer \
  pubmed-appnp-arch_appnp-relative2-epochs200.pgexplainer \
  cora-gat-arch_gat-relative2-epochs200.gat_attention \
  amazon_photo-gcn-seed-relative0.1-epochs200.gnnexplainer \
  amazon_photo-gat-arch_gat-relative0.1-epochs200.gnnexplainer \
  amazon_photo-graphsage-arch_graphsage-relative0.1-epochs200.gnnexplainer; do
  metadata_args+=(--metadata "outputs/explanations/${prefix}.metadata.json")
done
python -u -m gnn_rashomon.cli.reanalyze_explanation_dependencies "${metadata_args[@]}"

for dataset in cora citeseer pubmed; do
  python -u -m gnn_rashomon.cli.analyze_local_entanglement \
    --rewiring-metadata "outputs/graphs/${dataset}-local_entanglement-tau0p1-0p05-seed0.json" \
    --original-structure "outputs/metrics/${dataset}-gcn-seed-relative0.1-epochs200.structure_nodes.csv" \
    --rewired-structure "outputs/metrics/${dataset}-gcn-local_entanglement_tau0p1-relative2-epochs200.structure_nodes.csv" \
    --delta-table "outputs/metrics/${dataset}-local-entanglement-tau0p1-relative2-seed0.delta_multiplicity.csv" \
    --output-prefix "outputs/figures/${dataset}_local_entanglement_treatment_control"
done

python -u -m gnn_rashomon.cli.audit_fairness_concentration_dependencies \
  --adult-root "${ADULT_ROOT}" \
  --output-prefix "${ADULT_ROOT}/figures/fairness_concentration_dependency_audit" \
  --permutations 10000 \
  --seed 20260715

python -u -m gnn_rashomon.cli.build_dependency_structure_audit \
  --fairness-audit "${ADULT_ROOT}/figures/fairness_concentration_dependency_audit.csv" \
  --output-prefix outputs/figures/dependency_structure_audit \
  --permutations 5000 \
  --seed 20260715
