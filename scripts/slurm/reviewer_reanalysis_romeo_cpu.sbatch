#!/usr/bin/env bash
#SBATCH --job-name=reviewer_reanalysis
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -p romeo
#SBATCH --requeue
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err

# Reviewer reanalysis package (#2, #4, #5, #6). No new training.
#   #2+#6  validation-matched primary tables + Rashomon-definition ablation
#   #4     rewiring effect sizes + dose-response (graph-level)
#   #5     RQ3 equivalence / TOST on explanation regressions
#
# Optional node-level ΔD_i for #4 is a separate array:
#   sbatch scripts/slurm/materialize_rewiring_node_deltas_romeo_cpu_array.sbatch

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

echo "[$(date -Is)] #2+#6 membership sensitivity / definition ablation"
python -u -m gnn_rashomon.cli.build_rashomon_membership_sensitivity \
  --validation-accuracy-delta 0.01 \
  --validation-accuracy-delta 0.02 \
  --validation-accuracy-delta 0.05 \
  --primary-delta 0.02 \
  --output-dir outputs/rashomon_membership_sensitivity

echo "[$(date -Is)] #4 rewiring effect sizes + dose-response"
python -u -m gnn_rashomon.cli.build_rewiring_effect_sizes \
  --realizations-csv outputs/repeated_rewiring/repeated_rewiring_realizations.csv \
  --output-prefix outputs/rewiring_effect_sizes/rewiring_effect_sizes

if compgen -G "outputs/repeated_rewiring/random/node_deltas/*.csv" > /dev/null; then
  echo "[$(date -Is)] #4 node-level delta summaries (node_deltas present)"
  python -u -m gnn_rashomon.cli.build_rewiring_effect_sizes \
    --realizations-csv outputs/repeated_rewiring/repeated_rewiring_realizations.csv \
    --node-delta-glob 'outputs/repeated_rewiring/random/node_deltas/*.csv' \
    --output-prefix outputs/rewiring_effect_sizes/rewiring_effect_sizes
else
  echo "[$(date -Is)] #4 skipping node-level summaries (no node_deltas yet)"
fi

echo "[$(date -Is)] #5 RQ3 equivalence / TOST"
python -u -m gnn_rashomon.cli.run_equivalence_tests \
  --equivalence-margin 0.10 \
  --output-prefix outputs/equivalence/rq3_equivalence

echo "[$(date -Is)] done"
printf 'Wrote:\n'
printf '  outputs/rashomon_membership_sensitivity/\n'
printf '  outputs/rewiring_effect_sizes/\n'
printf '  outputs/equivalence/\n'
