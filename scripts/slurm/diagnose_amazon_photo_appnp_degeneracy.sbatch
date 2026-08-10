#!/bin/bash
#SBATCH --job-name=gnn_amzphoto_appnp_diag
#SBATCH --output=.logs/gnn_amzphoto_appnp_diag_%j.out
#SBATCH --error=.logs/gnn_amzphoto_appnp_diag_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -p romeo

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"
export PYTHONUNBUFFERED=1
export PYTHONPATH=${PROJECT_DIR}/src:${PYTHONPATH:-}
export CUDA_VISIBLE_DEVICES=""
export NVIDIA_VISIBLE_DEVICES=""

module purge
module load release/24.10 GCC/13.2.0 OpenMPI/4.1.6 PyTorch/2.3.0

# Optional: source your local cache/env setup here
source ${PROJECT_DIR}/.venv-module/bin/activate

python - <<'PY'
import csv
import json
from collections import Counter
from pathlib import Path

import torch

ROOT = Path(".")
OUT = ROOT / "outputs" / "figures" / "amazon_photo_appnp_degeneracy_diagnostic.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

paths = sorted(ROOT.glob("outputs/rashomon_sets/amazon_photo-appnp-arch_appnp-*.json"))

def walk(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk(v)

def get_predictions(obj):
    for d in walk(obj):
        for key in ("predictions", "y_pred", "pred", "test_predictions"):
            val = d.get(key) if isinstance(d, dict) else None
            if isinstance(val, list) and val and all(isinstance(x, (int, float, str)) for x in val):
                return [str(x) for x in val]
    return None

def records_from(data):
    if not isinstance(data, dict):
        return []
    if isinstance(data.get("runs"), dict):
        retained_ids = set(data.get("retained_run_ids", []))
        out = []
        for run_id, rec in data["runs"].items():
            if isinstance(rec, dict):
                row = dict(rec)
                row["run_id"] = run_id
                row["retained"] = run_id in retained_ids
                out.append(row)
        return out
    for key in ("records", "runs", "models", "candidates", "all_records", "all_runs"):
        value = data.get(key)
        if isinstance(value, list):
            return [x for x in value if isinstance(x, dict)]
    return []

def load_probability_predictions(rec):
    path = rec.get("probability_path")
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        p = ROOT / path
    if not p.exists():
        return None
    obj = torch.load(p, map_location="cpu")
    if isinstance(obj, dict):
        for key in ("probabilities", "probs", "logits", "predictions"):
            if key in obj:
                obj = obj[key]
                break
    tensor = torch.as_tensor(obj)
    if tensor.ndim == 1:
        return [str(int(x)) for x in tensor.tolist()]
    return [str(int(x)) for x in tensor.argmax(dim=-1).tolist()]

rows = []
for path in paths:
    with path.open() as f:
        data = json.load(f)

    records = records_from(data)
    pred_vectors = []
    class_counters = []
    for rec in records:
        pred = load_probability_predictions(rec) or get_predictions(rec)
        if pred is None:
            continue
        pred_vectors.append(pred)
        class_counters.append(Counter(pred))

    unique_vectors = {tuple(p) for p in pred_vectors}
    majority_labels = [c.most_common(1)[0][0] for c in class_counters if c]
    majority_counts = Counter(majority_labels)
    rows.append({
        "file": str(path),
        "records": len(records),
        "prediction_vectors_found": len(pred_vectors),
        "unique_prediction_vectors": len(unique_vectors),
        "majority_label_histogram": json.dumps(dict(majority_counts), sort_keys=True),
    })

with OUT.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "file", "records", "prediction_vectors_found",
        "unique_prediction_vectors", "majority_label_histogram",
    ])
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote APPNP degeneracy diagnostic to {OUT}")
print(f"rows={len(rows)}")
PY
