from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from gnn_rashomon.analysis.equivalence import equivalence_rows_from_regression

DEFAULT_REGRESSIONS = [
    "outputs/explanations/cora-gcn-seed-relative0.1-epochs200.gnnexplainer.regression.csv",
    "outputs/explanations/cora-gat-arch_gat-relative2-epochs200.gnnexplainer.regression.csv",
    "outputs/explanations/cora-appnp-arch_appnp-relative2-epochs200.gnnexplainer.regression.csv",
    "outputs/explanations/pubmed-gat-arch_gat-relative2-epochs200.gnnexplainer.regression.csv",
    "outputs/explanations/pubmed-appnp-arch_appnp-relative2-epochs200.gnnexplainer.regression.csv",
    "outputs/explanations/amazon_photo-gcn-seed-relative0.1-epochs200.gnnexplainer.regression.csv",
    "outputs/explanations/amazon_photo-gat-arch_gat-relative0.1-epochs200.gnnexplainer.regression.csv",
    "outputs/explanations/amazon_photo-graphsage-arch_graphsage-relative0.1-epochs200.gnnexplainer.regression.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Equivalence (TOST) analysis for explanation-instability regressions, "
            "testing whether multiplicity effects are practically near zero."
        )
    )
    parser.add_argument("--regression-csv", action="append", default=[])
    parser.add_argument("--term", default="rashomon_capacity")
    parser.add_argument(
        "--equivalence-margin",
        type=float,
        default=0.10,
        help="Smallest effect size of interest for |beta| on the Jaccard scale.",
    )
    parser.add_argument(
        "--output-prefix",
        default="outputs/equivalence/rq3_equivalence",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = [Path(path) for path in (args.regression_csv or DEFAULT_REGRESSIONS)]
    rows: list[pd.DataFrame] = []
    for path in paths:
        if not path.exists():
            print(f"Skipping missing regression: {path}")
            continue
        regression = pd.read_csv(path)
        label = path.name.replace(".regression.csv", "")
        rows.append(
            equivalence_rows_from_regression(
                regression,
                term=args.term,
                equivalence_margin=args.equivalence_margin,
                label=label,
            )
        )
    if not rows:
        raise SystemExit("No regression tables were processed.")
    table = pd.concat(rows, ignore_index=True)
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = prefix.with_suffix(".csv")
    json_path = prefix.with_suffix(".summary.json")
    table.to_csv(csv_path, index=False)
    summary = {
        "term": args.term,
        "equivalence_margin": args.equivalence_margin,
        "n_models": int(len(table)),
        "n_equivalent_at_0_05": int(table["equivalent_at_0_05"].sum()),
        "n_ci_inside_interval": int(table["ci_inside_equivalence_interval"].sum()),
        "output_csv": str(csv_path),
        "interpretation": (
            "equivalent_at_0_05=True supports the stronger claim that effects large enough "
            f"for |beta|>={args.equivalence_margin} are inconsistent with the data at alpha=0.05. "
            "Failure to reject TOST remains 'not shown equivalent', not positive evidence of a large effect."
        ),
    }
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
