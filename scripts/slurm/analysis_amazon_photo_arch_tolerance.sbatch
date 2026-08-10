#!/bin/bash
#SBATCH --job-name=gnn_amzphoto_arch_tol
#SBATCH --output=.logs/gnn_amzphoto_arch_tol_%j.out
#SBATCH --error=.logs/gnn_amzphoto_arch_tol_%j.err
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

python - <<'PY'
import csv
import json
import math
from pathlib import Path

ROOT = Path(".")
OUT = ROOT / "outputs" / "figures" / "amazon_photo_architecture_tolerance_audit.csv"
OUT.parent.mkdir(parents=True, exist_ok=True)

patterns = [
    "outputs/rashomon_sets/amazon_photo-*-arch_*-relative0.1-epochs200.json",
    "outputs/rashomon_sets/amazon_photo-*-arch_*-relative2-epochs200.json",
    "outputs/rashomon_sets/amazon_photo-gcn-*-relative0.1-epochs200.json",
    "outputs/rashomon_sets/amazon_photo-gcn-*-relative2-epochs200.json",
]

def as_float(x):
    try:
        if x is None:
            return None
        y = float(x)
        if math.isnan(y):
            return None
        return y
    except Exception:
        return None

def walk(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from walk(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk(v)

def first_number(d, keys):
    for k in keys:
        if k in d:
            v = as_float(d[k])
            if v is not None:
                return v
    return None

def records_from(data):
    if isinstance(data, dict) and isinstance(data.get("runs"), dict):
        retained_ids = set(data.get("retained_run_ids", []))
        out = []
        for run_id, rec in data["runs"].items():
            if isinstance(rec, dict):
                row = dict(rec)
                row["run_id"] = run_id
                row["retained"] = run_id in retained_ids
                out.append(row)
        return out
    candidates = []
    for key in ("records", "runs", "models", "candidates", "all_records", "all_runs"):
        value = data.get(key) if isinstance(data, dict) else None
        if isinstance(value, list):
            candidates.extend([x for x in value if isinstance(x, dict)])
    if candidates:
        return candidates
    # Fallback: use dicts that look like per-model records.
    out = []
    for d in walk(data):
        if any(k in d for k in ("loss", "val_loss", "test_loss", "accuracy", "test_accuracy", "retained")):
            out.append(d)
    return out

def retained_flag(rec):
    for k in ("retained", "is_retained", "in_rashomon"):
        if k in rec:
            return bool(rec[k])
    loss = first_number(rec, ("loss", "val_loss", "validation_loss", "test_loss", "train_loss", "risk"))
    threshold = first_number(rec, ("threshold", "rashomon_threshold"))
    return bool(loss is not None and threshold is not None and loss <= threshold)

rows = []
seen = set()
for pat in patterns:
    for path in sorted(ROOT.glob(pat)):
        if path in seen:
            continue
        seen.add(path)
        with path.open() as f:
            data = json.load(f)

        name = path.stem
        parts = name.split("-")
        model = parts[1] if len(parts) > 1 else "unknown"
        tolerance = "relative0.1" if "relative0.1" in name else ("relative2" if "relative2" in name else "unknown")

        top = data if isinstance(data, dict) else {}
        baseline_loss = first_number(top, ("baseline_loss", "best_loss", "reference_loss", "loss_baseline"))
        threshold = first_number(top, ("threshold", "rashomon_threshold", "loss_threshold"))

        records = records_from(data)
        generated = int(top.get("candidate_count", len(records)))
        retained = [r for r in records if retained_flag(r)]
        retained_count = int(top.get("retained_count", len(retained)))

        retained_losses = [first_number(r, ("loss", "val_loss", "validation_loss", "test_loss", "train_loss", "risk")) for r in retained]
        retained_losses = [x for x in retained_losses if x is not None]
        retained_accs = [first_number(r, ("accuracy", "acc", "test_accuracy", "val_accuracy", "validation_accuracy")) for r in retained]
        retained_accs = [x for x in retained_accs if x is not None]

        disagreement = first_number(top, ("disagreement_fraction", "disagreement", "node_disagreement_fraction"))
        if disagreement is None:
            for d in walk(data):
                disagreement = first_number(d, ("disagreement_fraction", "disagreement", "node_disagreement_fraction"))
                if disagreement is not None:
                    break

        rows.append({
            "file": str(path),
            "model": model,
            "tolerance": tolerance,
            "baseline_loss": baseline_loss,
            "threshold": threshold,
            "generated": generated,
            "retained": retained_count,
            "retained_loss_min": min(retained_losses) if retained_losses else None,
            "retained_loss_max": max(retained_losses) if retained_losses else None,
            "retained_accuracy_min": min(retained_accs) if retained_accs else None,
            "retained_accuracy_max": max(retained_accs) if retained_accs else None,
            "disagreement": disagreement,
        })

with OUT.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "file", "model", "tolerance", "baseline_loss", "threshold",
        "generated", "retained", "retained_loss_min", "retained_loss_max",
        "retained_accuracy_min", "retained_accuracy_max", "disagreement",
    ])
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote Amazon-Photo architecture tolerance audit to {OUT}")
print(f"rows={len(rows)}")
PY
