#!/usr/bin/env bash
#SBATCH --job-name=gnn_release_manifest
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -p romeo
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
MANIFEST_PATH="${MANIFEST_PATH:-outputs/metadata/release_manifest.json}"

if [[ -z "${ADULT_ROOT:-}" ]]; then
  ADULT_ROOT="${PROJECT_DIR}/outputs/release/adult_corrected"
  if [[ ! -d "${ADULT_ROOT}" ]]; then
    shopt -s nullglob
    adult_candidates=("${PROJECT_DIR}"/outputs/release/adult_corrected_*)
    shopt -u nullglob
    if (( ${#adult_candidates[@]} > 0 )); then
      ADULT_ROOT="${adult_candidates[${#adult_candidates[@]} - 1]}"
    fi
  fi
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES=""
export MANIFEST_PATH

module purge
module load release/24.10 GCC/13.2.0 OpenMPI/4.1.6 PyTorch/2.3.0
# Optional: source your local cache/env setup here
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv/bin/activate"

[[ -f "${ADULT_ROOT}/metadata/adult_corrected_graph_audit.json" ]] || {
  echo "Missing corrected Adult graph audit under ${ADULT_ROOT}" >&2
  exit 2
}
for accounting_artifact in \
  "${ADULT_ROOT}/metadata/adult_corrected_slurm_accounting.tsv" \
  "${ADULT_ROOT}/metadata/adult_corrected_slurm_job_ids.txt"; do
  [[ -s "${accounting_artifact}" ]] || {
    echo "Missing or empty accounting artifact: ${accounting_artifact}" >&2
    exit 2
  }
done
for source_manifest in \
  outputs/metadata/stage1_repro_manifest.json \
  outputs/metadata/next_steps_repro_manifest.json \
  outputs/metadata/reviewer_extensions_final_manifest.json; do
  [[ -f "${source_manifest}" ]] || { echo "Missing ${source_manifest}" >&2; exit 2; }
done

status=$(git status --short)
if [[ -n "${status}" ]]; then
  echo "Release manifest requires a clean git worktree:" >&2
  echo "${status}" >&2
  exit 2
fi

python -u -m gnn_rashomon.cli.build_repro_manifest \
  --output "${MANIFEST_PATH}" \
  --artifact-manifest outputs/metadata/stage1_repro_manifest.json \
  --artifact-manifest outputs/metadata/next_steps_repro_manifest.json \
  --artifact-manifest outputs/metadata/reviewer_extensions_final_manifest.json \
  --exclude-manifest-artifact-substring adult \
  --exclude-manifest-artifact-substring rewired_random \
  --exclude-manifest-artifact-substring homophily_targeted \
  --exclude-manifest-artifact-substring local_entanglement \
  --artifact "${ADULT_ROOT}/**/*.csv" \
  --artifact "${ADULT_ROOT}/**/*.tsv" \
  --artifact "${ADULT_ROOT}/**/*.txt" \
  --artifact "${ADULT_ROOT}/**/*.json" \
  --artifact "${ADULT_ROOT}/**/*.pt" \
  --artifact 'outputs/metadata/stage1_repro_manifest.json' \
  --artifact 'outputs/metadata/next_steps_repro_manifest.json' \
  --artifact 'outputs/metadata/reviewer_extensions_final_manifest.json' \
  --artifact 'outputs/figures/dependency_structure_audit.claims.csv' \
  --artifact 'outputs/figures/dependency_structure_audit.explanations.csv' \
  --artifact 'outputs/figures/dependency_structure_audit.summary.json' \
  --artifact 'outputs/explanations/*.regression.csv' \
  --artifact 'outputs/repeated_rewiring/*.csv' \
  --artifact 'outputs/repeated_rewiring/*.json' \
  --artifact 'outputs/repeated_rewiring/**/realizations/*.json' \
  --artifact 'outputs/repeated_rewiring/**/graphs/*.json' \
  --artifact 'outputs/repeated_rewiring/**/graphs/*.pt' \
  --artifact 'outputs/performance_sensitivity/*.csv' \
  --artifact 'outputs/performance_sensitivity/*.json' \
  --artifact 'outputs/performance_sensitivity/rashomon_sets/*.json' \
  --artifact 'outputs/set_size_robustness/*.csv' \
  --artifact 'outputs/set_size_robustness/*.json' \
  --artifact 'outputs/budget_matched_fairness/*.csv' \
  --artifact 'outputs/budget_matched_fairness/*.json' \
  --artifact 'outputs/non_gcn_rewiring/**/*.csv' \
  --artifact 'outputs/non_gcn_rewiring/**/*.json' \
  --artifact 'outputs/canonical_pubmed_gat/**/*.json' \
  --artifact 'outputs/canonical_pubmed_gat/**/*.pt' \
  --artifact 'outputs/architecture_feature_audit/*.csv' \
  --artifact 'outputs/architecture_feature_audit/*.json' \
  --artifact 'outputs/rashomon_sets/amazon_photo-gcn-hyperparameter-relative2-epochs200.json' \
  --artifact '../rashomon.tex' \
  --artifact '../references.bib' \
  --log-glob '.logs/gnn_adult_fix_*.out' \
  --log-glob '.logs/gnn_adult_fix_*.err' \
  --log-glob '.logs/gnn_release_acct_*.out' \
  --log-glob '.logs/gnn_release_acct_*.err' \
  --log-glob '.logs/gnn_dep_audit_*.out' \
  --log-glob '.logs/gnn_dep_audit_*.err' \
  --log-glob '.logs/gnn_repeat_*.out' \
  --log-glob '.logs/gnn_repeat_*.err' \
  --log-glob '.logs/gnn_set_size_*.out' \
  --log-glob '.logs/gnn_set_size_*.err' \
  --log-glob '.logs/set_size_summary_*.out' \
  --log-glob '.logs/set_size_summary_*.err' \
  --log-glob '.logs/gnn_budget_fair_*.out' \
  --log-glob '.logs/gnn_budget_fair_*.err' \
  --log-glob '.logs/gat_*rw*.out' \
  --log-glob '.logs/gat_*rw*.err' \
  --log-glob '.logs/gat_canon_seed_*.out' \
  --log-glob '.logs/gat_canon_seed_*.err' \
  --log-glob '.logs/feature_repr_audit_*.out' \
  --log-glob '.logs/feature_repr_audit_*.err'

python - <<'PY'
import json
import os
from pathlib import Path

path = Path(os.environ["MANIFEST_PATH"])
manifest = json.loads(path.read_text(encoding="utf-8"))
missing = manifest.get("missing_manifest_artifacts", [])
if missing:
    preview = "\n".join(missing[:20])
    raise SystemExit(f"Release manifest has {len(missing)} missing imported artifacts:\n{preview}")
print(
    "Release manifest validated: "
    f"artifacts={len(manifest['artifacts'])} logs={len(manifest['logs'])}"
)
PY
