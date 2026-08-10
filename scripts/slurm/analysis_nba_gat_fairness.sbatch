#!/usr/bin/env bash
#SBATCH --job-name=gnn_nba_gat_fair
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH -p alpha
#SBATCH --requeue
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTHONPATH=${PROJECT_DIR}/src:${PYTHONPATH:-}
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
module purge
module load release/24.10 GCC/13.2.0 OpenMPI/4.1.6 PyTorch/2.3.0
# Optional: source your local cache/env setup here
PROJECT_DIR=${PROJECT_DIR}
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv/bin/activate"

METRICS=(demographic_parity_gap equalized_odds_gap degree_disparity)
for set_type in seed hyperparameter; do
  if [[ "${set_type}" == seed ]]; then
    epsilon=0.10; suffix=relative0.1; dedupe=(); run_type=seed_nba_gat_revised; expected=50
  else
    epsilon=0.20; suffix=relative0.2; dedupe=(--no-deduplicate-seeds); run_type=hyperparameter_nba_gat_revised; expected=36
  fi
  python -u -m gnn_rashomon.cli.audit_rashomon_health \
    --dataset nba --architecture gat --rashomon-type "${run_type}" \
    --max-epochs 300 --expected-candidates "${expected}" \
    --tolerance-mode relative --epsilon "${epsilon}" \
    --output-prefix "outputs/figures/nba_gat_${set_type}_revised_health"
  python -u -m gnn_rashomon.cli.generate_rashomon \
    --dataset nba --architecture gat --set-type "${set_type}" \
    --rashomon-type "${run_type}" --max-epochs 300 \
    --tolerance-mode relative --epsilon "${epsilon}" "${dedupe[@]}"
  SET_ID="nba-gat-${set_type}-${suffix}-epochs300"
  python -u -m gnn_rashomon.cli.compute_multiplicity \
    "outputs/rashomon_sets/${SET_ID}.json" --capacity-method chunked --capacity-chunk-size 16
  python -u -m gnn_rashomon.cli.run_fairness \
    --dataset nba --rashomon-set "outputs/rashomon_sets/${SET_ID}.json" \
    --multiplicity-table "outputs/metrics/${SET_ID}.multiplicity_nodes.csv" \
    --sensitive-attribute country
  for metric in "${METRICS[@]}"; do
    python -u -m gnn_rashomon.cli.build_pareto \
      --mode fairness \
      --fairness-table "outputs/metrics/${SET_ID}.fairness_models.csv" \
      --fairness-metric "${metric}" \
      --output-prefix "outputs/figures/${SET_ID}.fairness_${metric}.pareto"
  done
done
