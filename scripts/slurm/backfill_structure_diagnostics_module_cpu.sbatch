#!/usr/bin/env bash
#SBATCH --job-name=gnn_struct_diag
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
PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv-module/bin/activate"

python -u - <<'PY'
import json
import re
import subprocess
import sys
from pathlib import Path

def graph_path_for_set(set_id):
    random_match = re.search(r"rewired_random_(0p\d+)", set_id)
    if random_match:
        return Path("outputs/graphs") / f"cora-random-{random_match.group(1)}-seed0.pt"
    homophily_match = re.search(r"homophily_targeted_(0p\d+)", set_id)
    if homophily_match:
        target = homophily_match.group(1)
        return Path("outputs/graphs") / f"cora-homophily_targeted-target{target}-0p05-seed0.pt"
    return None

root = Path("outputs/rashomon_sets")
for path in sorted(root.glob("*.json")):
    payload = json.loads(path.read_text(encoding="utf-8"))
    set_id = payload["set_id"]
    dataset = payload["dataset"]
    graph_path = graph_path_for_set(set_id)
    multiplicity = Path("outputs/metrics") / f"{set_id}.multiplicity_nodes.csv"
    if not multiplicity.exists():
        print(f"Skipping {set_id}; missing {multiplicity}", flush=True)
        continue
    command = [
        sys.executable,
        "-u",
        "-m",
        "gnn_rashomon.cli.compute_structure",
        "--dataset",
        dataset,
        "--rashomon-set",
        str(path),
        "--multiplicity-table",
        str(multiplicity),
        "--include-community",
    ]
    if graph_path is not None:
        if not graph_path.exists():
            print(f"Skipping {set_id}; missing graph artifact {graph_path}", flush=True)
            continue
        command.extend(["--graph-path", str(graph_path)])
    print(f"Backfilling structural diagnostics for {set_id}", flush=True)
    subprocess.check_call(command)

reports = [
    ("*-gcn-seed-relative0.1-epochs200", "seed_relative0.1"),
    ("*-gcn-hyperparameter-relative2-epochs200", "hyperparameter_relative2"),
    ("*-mlp-seed-relative0.1-epochs200", "mlp_seed_relative0.1"),
    ("*-mlp-hyperparameter-relative2-epochs200", "mlp_hyperparameter_relative2"),
    ("*-topology_only-seed-relative0.1-epochs200", "topology_only_seed_relative0.1"),
    ("*-topology_only-hyperparameter-relative2-epochs200", "topology_only_hyperparameter_relative2"),
]
for pattern, report in reports:
    print(f"Rebuilding aggregate report {report}", flush=True)
    subprocess.call(
        [
            sys.executable,
            "-u",
            "-m",
            "gnn_rashomon.cli.build_report",
            "--set-pattern",
            pattern,
            "--report-name",
            report,
        ]
    )
    subprocess.call(
        [
            sys.executable,
            "-u",
            "-m",
            "gnn_rashomon.cli.plot_report",
            "--report-name",
            report,
        ]
    )
PY

echo "Structural diagnostics backfill complete"
