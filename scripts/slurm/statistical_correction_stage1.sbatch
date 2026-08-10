#!/bin/bash
#SBATCH --job-name=gnn_stage1_padjust
#SBATCH --output=.logs/gnn_stage1_padjust_%j.out
#SBATCH --error=.logs/gnn_stage1_padjust_%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$PWD}"

python - <<'PY'
import csv
import json
import math
import re
from pathlib import Path

ROOT = Path(".")
OUT_CSV = ROOT / "outputs" / "figures" / "stage1_multiple_testing_correction.csv"
OUT_JSON = ROOT / "outputs" / "figures" / "stage1_multiple_testing_correction_summary.json"
OUT_CSV.parent.mkdir(parents=True, exist_ok=True)

P_PATTERNS = [
    re.compile(r"(^|_)(p|pvalue|p_value|pval|p_val)($|_)"),
    re.compile(r"(^|_)(Pr\(>\|t\|\)|Pr\(>\|z\|\))($|_)"),
]

def is_p_col(name):
    low = name.strip().lower()
    return any(p.search(low) for p in P_PATTERNS)

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

def bh_adjust(pvals):
    n = len(pvals)
    indexed = sorted(enumerate(pvals), key=lambda x: x[1])
    out = [None] * n
    running = 1.0
    for rank_from_end, (idx, p) in enumerate(reversed(indexed), start=1):
        rank = n - rank_from_end + 1
        running = min(running, p * n / rank)
        out[idx] = min(running, 1.0)
    return out

rows = []
for path in sorted((ROOT / "outputs" / "metrics").glob("*.csv")):
    try:
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                continue
            p_cols = [c for c in reader.fieldnames if is_p_col(c)]
            if not p_cols:
                continue
            for i, row in enumerate(reader, start=1):
                for col in p_cols:
                    p = as_float(row.get(col))
                    if p is None or p < 0 or p > 1:
                        continue
                    family = path.stem
                    rows.append({
                        "family": family,
                        "file": str(path),
                        "row": i,
                        "p_column": col,
                        "raw_p": p,
                    })
    except Exception as exc:
        print(f"Skipping {path}: {exc}")

for family in sorted({r["family"] for r in rows}):
    fam_rows = [r for r in rows if r["family"] == family]
    pvals = [r["raw_p"] for r in fam_rows]
    bh = bh_adjust(pvals)
    n = len(pvals)
    for r, q in zip(fam_rows, bh):
        r["bh_fdr_p"] = q
        r["bonferroni_p"] = min(r["raw_p"] * n, 1.0)
        r["significant_raw_0.05"] = r["raw_p"] < 0.05
        r["significant_bh_0.05"] = q < 0.05
        r["significant_bonferroni_0.05"] = r["bonferroni_p"] < 0.05
        r["family_size"] = n

fieldnames = [
    "family", "family_size", "file", "row", "p_column", "raw_p",
    "bh_fdr_p", "bonferroni_p",
    "significant_raw_0.05", "significant_bh_0.05", "significant_bonferroni_0.05",
]
with OUT_CSV.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

summary = {
    "tests": len(rows),
    "families": len({r["family"] for r in rows}),
    "raw_significant_0.05": sum(bool(r.get("significant_raw_0.05")) for r in rows),
    "bh_significant_0.05": sum(bool(r.get("significant_bh_0.05")) for r in rows),
    "bonferroni_significant_0.05": sum(bool(r.get("significant_bonferroni_0.05")) for r in rows),
    "output_csv": str(OUT_CSV),
}
with OUT_JSON.open("w") as f:
    json.dump(summary, f, indent=2, sort_keys=True)

print(f"Wrote multiple-testing correction table to {OUT_CSV}")
print(f"Wrote summary to {OUT_JSON}")
print(json.dumps(summary, indent=2, sort_keys=True))
PY
