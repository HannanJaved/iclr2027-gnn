#!/usr/bin/env bash
#SBATCH --job-name=cf_retrain_analyze
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -p romeo
#SBATCH --requeue
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err

# Build Rashomon sets + multiplicity for all counterfactual retraining conditions,
# then summarize R(G') versus R(G).

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

python - <<'PY'
from gnn_rashomon.analysis.counterfactual_retraining import (
    CONDITIONS,
    DATASETS,
    EPSILON,
    GRAPH_SEEDS,
    MAX_EPOCHS,
    OUTPUT_ROOT,
    graph_path_for,
    set_type_for,
)
import subprocess
import sys

for dataset in DATASETS:
    for condition in CONDITIONS:
        for graph_seed in GRAPH_SEEDS:
            set_type = set_type_for(condition, graph_seed)
            rashomon_type = set_type
            graph_path = graph_path_for(dataset, condition, graph_seed)
            set_path = (
                OUTPUT_ROOT
                / "rashomon_sets"
                / f"{dataset}-gcn-{set_type}-relative{EPSILON:g}-epochs{MAX_EPOCHS}.json"
            )
            cmd_generate = [
                sys.executable,
                "-u",
                "-m",
                "gnn_rashomon.cli.generate_rashomon",
                "--dataset",
                dataset,
                "--architecture",
                "gcn",
                "--set-type",
                set_type,
                "--rashomon-type",
                rashomon_type,
                "--output-dir",
                str(OUTPUT_ROOT),
                "--max-epochs",
                str(MAX_EPOCHS),
                "--tolerance-mode",
                "relative",
                "--epsilon",
                str(EPSILON),
            ]
            print("Running:", " ".join(cmd_generate), flush=True)
            subprocess.run(cmd_generate, check=True)
            cmd_mult = [
                sys.executable,
                "-u",
                "-m",
                "gnn_rashomon.cli.compute_multiplicity",
                str(set_path),
                "--graph-path",
                str(graph_path),
                "--output-dir",
                str(OUTPUT_ROOT),
            ]
            print("Running:", " ".join(cmd_mult), flush=True)
            subprocess.run(cmd_mult, check=True)
PY

python -u -m gnn_rashomon.cli.summarize_counterfactual_retraining \
  --output-root outputs/counterfactual_retraining \
  --baseline-root outputs \
  --output-prefix outputs/counterfactual_retraining/cf_retrain_summary
