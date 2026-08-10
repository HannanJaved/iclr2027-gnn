#!/bin/bash
#SBATCH --job-name=gnn_amzphoto_arch_xai
#SBATCH --output=.logs/gnn_amzphoto_arch_xai_%j.out
#SBATCH --error=.logs/gnn_amzphoto_arch_xai_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

python - <<'PY'
import importlib.util
import subprocess
import sys

models = {
    "gat": "outputs/rashomon_sets/amazon_photo-gat-arch_gat-relative0.1-epochs200.json",
    "graphsage": "outputs/rashomon_sets/amazon_photo-graphsage-arch_graphsage-relative0.1-epochs200.json",
    "appnp": "outputs/rashomon_sets/amazon_photo-appnp-arch_appnp-relative0.1-epochs200.json",
}
modules = [
    "gnn_rashomon.cli.run_explanation_analysis",
    "gnn_rashomon.cli.explanation_analysis",
    "gnn_rashomon.cli.analyze_explanations",
]

module = next((m for m in modules if importlib.util.find_spec(m) is not None), None)
if module is None:
    print("No known explanation CLI module found. Skipping Amazon-Photo architecture explanation job cleanly.")
    raise SystemExit(0)

for model, rashomon_set in models.items():
    cmd = [
        sys.executable, "-m", module,
        "--dataset", "amazon_photo",
        "--model", model,
        "--rashomon-set", rashomon_set,
        "--output-prefix", f"outputs/figures/amazon_photo_{model}",
    ]
    print("Running", " ".join(cmd), flush=True)
    code = subprocess.call(cmd)
    if code != 0:
        raise SystemExit(code)
PY
