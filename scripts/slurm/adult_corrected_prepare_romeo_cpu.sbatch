#!/usr/bin/env bash
#SBATCH --job-name=gnn_adult_fix_prep
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH -p romeo
#SBATCH --output=.logs/%x_%j.out
#SBATCH --error=.logs/%x_%j.err

set -euo pipefail

PROJECT_DIR=${PROJECT_DIR}
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_DIR}/outputs/release/adult_corrected}"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES=""

module purge
module load release/24.10 GCC/13.2.0 OpenMPI/4.1.6 PyTorch/2.3.0
# Optional: source your local cache/env setup here
cd "${PROJECT_DIR}"
source "${PROJECT_DIR}/.venv/bin/activate"

mkdir -p "${OUTPUT_ROOT}/metadata"
export OUTPUT_ROOT

python -u - <<'PY'
import json
import os
from pathlib import Path

import torch

from gnn_rashomon.cli.config import load_config
from gnn_rashomon.data.loaders import load_graph

project = Path.cwd()
output_root = Path(os.environ["OUTPUT_ROOT"])
config = load_config(
    [
        "dataset=adult",
        "dataset_config.source=local",
        "dataset_config.csv_path=data/Adult/adult.csv",
        "dataset_config.max_nodes=5000",
        "dataset_config.k_neighbors=10",
        "dataset_config.use_cache=true",
        "dataset_config.include_sensitive_feature=false",
    ],
    project / "configs",
)

graph = load_graph(
    "adult",
    root=str(project / config["dataset_config"]["root"]),
    dataset_config=config["dataset_config"],
)
feature_names = [str(name) for name in getattr(graph.data, "feature_names", [])]
sensitive_columns = [
    name for name in feature_names if name.lower() == "sex" or name.lower().startswith("sex_")
]
if sensitive_columns:
    raise SystemExit(f"Adult audit failed: sensitive feature columns remain: {sensitive_columns}")
if int(graph.data.num_nodes) != 5000:
    raise SystemExit(f"Adult audit failed: expected 5000 nodes, got {graph.data.num_nodes}")
if "sex" not in graph.sensitive_attributes:
    raise SystemExit("Adult audit failed: sex is missing from sensitive attributes")

cache_path = project / config["dataset_config"]["root"] / "processed" / "adult_graph.pt"
cache = torch.load(cache_path, map_location="cpu", weights_only=False)
cache_config = cache.get("adult", {}).get("cache_config", {})
expected = {
    "max_nodes": 5000,
    "k_neighbors": 10,
    "include_sensitive_feature": False,
}
for key, value in expected.items():
    if cache_config.get(key) != value:
        raise SystemExit(
            f"Adult cache audit failed for {key}: {cache_config.get(key)!r} != {value!r}"
        )

audit = {
    "passed": True,
    "cache_path": str(cache_path),
    "cache_config": cache_config,
    "num_nodes": int(graph.data.num_nodes),
    "num_edges": int(graph.data.edge_index.shape[1]),
    "num_features": int(graph.data.num_node_features),
    "sensitive_feature_columns": sensitive_columns,
}
path = output_root / "metadata" / "adult_corrected_graph_audit.json"
path.write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
print(json.dumps(audit, indent=2, sort_keys=True))
print(f"Wrote {path}")
PY
