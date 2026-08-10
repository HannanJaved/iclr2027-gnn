#!/usr/bin/env bash
#SBATCH --job-name=feature_repr_audit
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH -p romeo
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR}
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES=""

module purge
module load release/24.10 GCC/13.2.0 OpenMPI/4.1.6 PyTorch/2.3.0
# Optional: source your local cache/env setup here
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv-module/bin/activate"

python -u -m gnn_rashomon.cli.audit_feature_representation \
  --rashomon-set outputs/rashomon_sets/cora-gcn-seed-relative0.1-epochs200.json \
  --rashomon-set outputs/rashomon_sets/cora-gat-arch_gat-relative2-epochs200.json \
  --rashomon-set outputs/rashomon_sets/cora-graphsage-arch_graphsage-relative2-epochs200.json \
  --rashomon-set outputs/rashomon_sets/cora-appnp-arch_appnp-relative2-epochs200.json \
  --rashomon-set outputs/rashomon_sets/pubmed-gcn-seed-relative0.1-epochs200.json \
  --rashomon-set outputs/rashomon_sets/pubmed-gat-arch_gat-relative2-epochs200.json \
  --rashomon-set outputs/rashomon_sets/pubmed-graphsage-arch_graphsage-relative2-epochs200.json \
  --rashomon-set outputs/rashomon_sets/pubmed-appnp-arch_appnp-relative2-epochs200.json \
  --rashomon-set outputs/canonical_pubmed_gat/rashomon_sets/pubmed-gat-canonical_seed-relative2-epochs200.json \
  --tolerance 1e-5 \
  --device cpu \
  --output-prefix outputs/architecture_feature_audit/citation_checkpoint_feature_representation
