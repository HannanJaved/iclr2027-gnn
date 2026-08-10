#!/bin/bash
#SBATCH --job-name=gnn_middle_link
#SBATCH --output=.logs/gnn_middle_link_%j.out
#SBATCH --error=.logs/gnn_middle_link_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

python - <<'PY'
import csv
import json
import math
from pathlib import Path

ROOT = Path(".")
OUT = ROOT / "outputs" / "figures" / "stage1_causal_middle_link_audit.csv"
SUMMARY = ROOT / "outputs" / "figures" / "stage1_causal_middle_link_audit_summary.json"
OUT.parent.mkdir(parents=True, exist_ok=True)

STRUCTURE_COLS = [
    "degree", "local_homophily", "homophily", "neighborhood_label_entropy",
    "label_entropy", "boundary_distance", "distance_to_boundary",
]
MIDDLE_COLS = [
    "embedding_variance", "embedding_var", "logit_variance", "logit_var",
    "probability_variance", "probability_var", "message_variance", "message_var",
]

def as_float(x):
    try:
        if x is None or x == "":
            return None
        y = float(x)
        if math.isnan(y):
            return None
        return y
    except Exception:
        return None

def mean(xs):
    return sum(xs) / len(xs) if xs else None

def spearman(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if x is not None and y is not None]
    if len(pairs) < 3:
        return None
    xvals, yvals = zip(*pairs)
    def ranks(vals):
        order = sorted(enumerate(vals), key=lambda p: p[1])
        out = [0.0] * len(vals)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and order[j + 1][1] == order[i][1]:
                j += 1
            rank = (i + j + 2) / 2.0
            for k in range(i, j + 1):
                out[order[k][0]] = rank
            i = j + 1
        return out
    rx, ry = ranks(xvals), ranks(yvals)
    mx, my = mean(rx), mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    denx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    deny = math.sqrt(sum((b - my) ** 2 for b in ry))
    if denx == 0 or deny == 0:
        return None
    return num / (denx * deny)

def numeric_columns(rows, cols):
    out = {}
    for col in cols:
        vals = [as_float(r.get(col)) for r in rows]
        if sum(v is not None for v in vals) >= 3:
            out[col] = vals
    return out

result_rows = []
candidate_files = sorted((ROOT / "outputs" / "metrics").glob("*.structure_nodes.csv"))
candidate_files += sorted((ROOT / "outputs" / "metrics").glob("*middle_link*.csv"))

for path in candidate_files:
    try:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            continue
        structure = numeric_columns(rows, STRUCTURE_COLS)
        middle = numeric_columns(rows, MIDDLE_COLS)
        if not structure or not middle:
            continue
        for s_name, s_vals in structure.items():
            for m_name, m_vals in middle.items():
                rho = spearman(s_vals, m_vals)
                result_rows.append({
                    "file": str(path),
                    "structure_metric": s_name,
                    "middle_metric": m_name,
                    "spearman_rho": rho,
                    "n": sum(x is not None and y is not None for x, y in zip(s_vals, m_vals)),
                })
    except Exception as exc:
        print(f"Skipping {path}: {exc}")

with OUT.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "file", "structure_metric", "middle_metric", "spearman_rho", "n"
    ])
    writer.writeheader()
    writer.writerows(result_rows)

summary = {
    "candidate_files": len(candidate_files),
    "tested_relationships": len(result_rows),
    "output_csv": str(OUT),
    "note": "If tested_relationships is zero, no existing artifact exposed embedding/logit/message variance columns yet.",
}
with SUMMARY.open("w") as f:
    json.dump(summary, f, indent=2, sort_keys=True)

print(f"Wrote causal middle-link audit to {OUT}")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
