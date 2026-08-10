#!/usr/bin/env bash
#SBATCH --job-name=cf_retrain_seed
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH -p alpha
#SBATCH --requeue
#SBATCH --array=0-2249%20
#SBATCH --output=.logs/%x_%A_%a.out
#SBATCH --error=.logs/%x_%A_%a.err

# Counterfactual retraining (#1): retrain GCN seed ensembles on fixed counterfactual graphs.
# Layout: 3 datasets × 3 conditions × 5 graph seeds × 50 train seeds = 2250 tasks.
# Reuses existing repeated-rewiring graphs (canonical NormalizeFeatures path).

set -euo pipefail
export PYTHONUNBUFFERED=1

module load CUDA
# Optional: source your local cache/env setup here
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv/bin/activate"
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"

mapfile -t DECODED < <(
  python - <<'PY'
from gnn_rashomon.analysis.counterfactual_retraining import (
    decode_array_task,
    graph_path_for,
    rashomon_type_for,
)
import os

dataset, condition, graph_seed, train_seed = decode_array_task(int(os.environ["SLURM_ARRAY_TASK_ID"]))
graph_path = graph_path_for(dataset, condition, graph_seed)
print(dataset)
print(condition.condition_id)
print(graph_seed)
print(train_seed)
print(graph_path)
print(rashomon_type_for(condition, graph_seed))
PY
)

dataset="${DECODED[0]}"
condition_id="${DECODED[1]}"
graph_seed="${DECODED[2]}"
train_seed="${DECODED[3]}"
graph_path="${DECODED[4]}"
rashomon_type="${DECODED[5]}"

test -f "${graph_path}"

python -u -m gnn_rashomon.cli.train_baseline \
  dataset="${dataset}" \
  graph_path="${graph_path}" \
  rashomon_type="${rashomon_type}" \
  seed="${train_seed}" \
  paths.output_dir=outputs/counterfactual_retraining \
  trainer.device=cuda \
  trainer.max_epochs=200
