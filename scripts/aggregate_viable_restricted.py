#!/usr/bin/env python3
"""Summarize within-subset viable-restricted rewiring results.

Run after evaluate_viable_restricted_matched_struct.sbatch completes:
    python scripts/aggregate_viable_restricted.py

For each dataset x condition (matched / struct), reports two comparisons:

1. Full-set delta (K = n_total on both G and G'): should match Table 3's
   already-published Matched/Struct columns, as a sanity check on this
   independently written script.

2. Within-subset delta (K = n_viable on both G and G'): the like-for-like
   estimator, where set size cancels exactly because the SAME models are
   used for both the G and G' diameter in a given realization. This is the
   number that actually answers "does which-edges-move survive restricting
   to viable members," not the earlier (confounded) full-vs-restricted
   comparison.

3. Mean pairwise total variation, both full-set and within-subset, as a
   cross-check: Appendix A.4 already establishes mean TV is stable to
   within 0.0018 under common-K subsampling, so if TV shows the same
   pattern as diameter, the effect is not a K artifact; if it does not,
   the diameter reversal was K.

Realizations with fewer than 2 viable models cannot support a within-subset
comparison and are reported separately as undefined (matters most for
PubMed matched, and also CiteSeer matched at 14/20).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASETS = ["cora", "citeseer", "pubmed"]
CONDITIONS = ["matched", "struct"]


def bootstrap_ci(values: np.ndarray, samples: int = 10000, seed: int = 0) -> tuple[float, float]:
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(samples, values.size), replace=True)
    means = draws.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def summarize(values: pd.Series) -> dict[str, float]:
    clean = values.dropna().to_numpy()
    lo, hi = bootstrap_ci(clean)
    return {
        "n": int(clean.size),
        "mean": float(clean.mean()) if clean.size else float("nan"),
        "ci_low": lo,
        "ci_high": hi,
    }


def main() -> None:
    rows = []
    for condition in CONDITIONS:
        for dataset in DATASETS:
            out_dir = (
                PROJECT_ROOT
                / "outputs/rewiring_effect_sizes/viable_restricted_within_subset"
                / condition
            )
            paths = sorted(out_dir.glob(f"{dataset}-*seed*.json"))
            if not paths:
                print(f"[missing] {condition} {dataset}: no output files under {out_dir}")
                continue
            records = [json.loads(p.read_text()) for p in paths]
            df = pd.DataFrame(records)

            full_delta = summarize(df["delta_diameter_full"])
            viable_delta = summarize(df["delta_diameter_viable"])
            full_tv = summarize(df["delta_tv_full"])
            viable_tv = summarize(df["delta_tv_viable"])
            n_undefined = int(df["delta_diameter_viable"].isna().sum())

            rows.append(
                {
                    "condition": condition,
                    "dataset": dataset,
                    "n_realizations": len(df),
                    "n_viable_defined": viable_delta["n"],
                    "n_viable_undefined": n_undefined,
                    "full_set_delta_diameter": full_delta["mean"],
                    "full_set_delta_diameter_ci_low": full_delta["ci_low"],
                    "full_set_delta_diameter_ci_high": full_delta["ci_high"],
                    "within_subset_delta_diameter": viable_delta["mean"],
                    "within_subset_delta_diameter_ci_low": viable_delta["ci_low"],
                    "within_subset_delta_diameter_ci_high": viable_delta["ci_high"],
                    "full_set_delta_tv": full_tv["mean"],
                    "within_subset_delta_tv": viable_tv["mean"],
                    "within_subset_delta_tv_ci_low": viable_tv["ci_low"],
                    "within_subset_delta_tv_ci_high": viable_tv["ci_high"],
                }
            )

    summary = pd.DataFrame(rows)
    if summary.empty:
        print("No results found yet -- has the sbatch array run to completion?")
        return

    out_csv = PROJECT_ROOT / "outputs/rewiring_effect_sizes/viable_restricted_within_subset_summary.csv"
    summary.to_csv(out_csv, index=False)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    print(summary.to_string(index=False))

    print("\n--- like-for-like comparison (within-subset delta diameter) ---")
    for dataset in DATASETS:
        m = summary[(summary.condition == "matched") & (summary.dataset == dataset)]
        s = summary[(summary.condition == "struct") & (summary.dataset == dataset)]
        if m.empty or s.empty:
            continue
        m, s = m.iloc[0], s.iloc[0]
        print(
            f"{dataset}: matched={m.within_subset_delta_diameter:.4f} "
            f"[{m.within_subset_delta_diameter_ci_low:.4f},{m.within_subset_delta_diameter_ci_high:.4f}] "
            f"(n={m.n_viable_defined}/{m.n_realizations}) vs "
            f"struct={s.within_subset_delta_diameter:.4f} "
            f"[{s.within_subset_delta_diameter_ci_low:.4f},{s.within_subset_delta_diameter_ci_high:.4f}] "
            f"(n={s.n_viable_defined}/{s.n_realizations})  |  "
            f"TV cross-check: matched={m.within_subset_delta_tv:.5f} vs struct={s.within_subset_delta_tv:.5f}"
        )

    print(f"\nWrote {out_csv}")


if __name__ == "__main__":
    main()
