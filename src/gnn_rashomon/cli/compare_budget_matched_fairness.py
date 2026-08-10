from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from gnn_rashomon.analysis.budget_matched_fairness import (
    budget_matched_comparisons,
    summarize_budget_matched,
)
from gnn_rashomon.cli.compare_fairgnn_baseline import _infer_metric, _infer_set_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Downsample a retained Rashomon candidate pool to the baseline run count before "
            "validation-only fairness selection."
        )
    )
    parser.add_argument("--baseline-table", required=True)
    parser.add_argument("--candidate-models", nargs="+", required=True)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Defaults to the number of rows in the baseline table.",
    )
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--output-prefix", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = pd.read_csv(args.baseline_table)
    sample_size = args.sample_size or len(baseline)
    tables = []
    sources = []
    for index, raw_path in enumerate(args.candidate_models):
        path = Path(raw_path)
        candidates = pd.read_csv(path)
        metric = _infer_metric(path, candidates)
        set_id = _infer_set_id(path, candidates)
        tables.append(
            budget_matched_comparisons(
                baseline,
                candidates,
                metric=metric,
                sample_size=sample_size,
                draws=args.draws,
                seed=args.seed + index,
                set_id=set_id,
            )
        )
        sources.append(str(path))

    draws = pd.concat(tables, ignore_index=True)
    summary = summarize_budget_matched(draws)
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    draw_path = Path(f"{prefix}.draws.csv")
    summary_path = Path(f"{prefix}.summary.csv")
    metadata_path = Path(f"{prefix}.metadata.json")
    draws.to_csv(draw_path, index=False)
    summary.to_csv(summary_path, index=False)
    metadata = {
        "baseline_table": args.baseline_table,
        "candidate_model_tables": sources,
        "sample_size": sample_size,
        "draw_count_per_set_and_metric": args.draws,
        "seed": args.seed,
        "selection_uses_test_metrics": False,
        "test_metrics_are_not_loaded": True,
        "estimand": (
            "Retained-pool downsampling sensitivity. This standardizes the number of models "
            "available to the deployment selector; it does not equate training compute."
        ),
        "draw_table": str(draw_path),
        "summary_table": str(summary_path),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
