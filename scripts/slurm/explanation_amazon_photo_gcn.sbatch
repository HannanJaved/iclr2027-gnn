#!/bin/bash
#SBATCH --job-name=gnn_amzphoto_gcn_xai
#SBATCH --output=.logs/gnn_amzphoto_gcn_xai_%j.out
#SBATCH --error=.logs/gnn_amzphoto_gcn_xai_%j.err
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

python - <<'PY'
import importlib.util
import subprocess
import sys

candidates = [
    ("gnn_rashomon.cli.run_explanation_analysis", [
        "--dataset", "amazon_photo",
        "--model", "gcn",
        "--rashomon-set", "outputs/rashomon_sets/amazon_photo-gcn-seed-relative0.1-epochs200.json",
        "--output-prefix", "outputs/figures/amazon_photo_gcn",
    ]),
    ("gnn_rashomon.cli.explanation_analysis", [
        "--dataset", "amazon_photo",
        "--model", "gcn",
        "--rashomon-set", "outputs/rashomon_sets/amazon_photo-gcn-seed-relative0.1-epochs200.json",
        "--output-prefix", "outputs/figures/amazon_photo_gcn",
    ]),
    ("gnn_rashomon.cli.analyze_explanations", [
        "--dataset", "amazon_photo",
        "--model", "gcn",
        "--rashomon-set", "outputs/rashomon_sets/amazon_photo-gcn-seed-relative0.1-epochs200.json",
        "--output-prefix", "outputs/figures/amazon_photo_gcn",
    ]),
]

for module, args in candidates:
    if importlib.util.find_spec(module) is None:
        continue
    cmd = [sys.executable, "-m", module] + args
    print("Running", " ".join(cmd), flush=True)
    raise SystemExit(subprocess.call(cmd))

print("No known explanation CLI module found. Skipping Amazon-Photo GCN explanation job cleanly.")
PY
