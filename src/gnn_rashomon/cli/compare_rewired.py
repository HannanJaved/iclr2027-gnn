from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare original and rewired node multiplicity tables.")
    parser.add_argument("--original-table", required=True)
    parser.add_argument("--rewired-table", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    original = pd.read_csv(args.original_table).add_prefix("original_")
    rewired = pd.read_csv(args.rewired_table).add_prefix("rewired_")
    joined = original.merge(rewired, left_on="original_node_id", right_on="rewired_node_id")
    joined["delta_predictive_entropy"] = (
        joined["rewired_predictive_entropy"] - joined["original_predictive_entropy"]
    )
    joined["delta_variation_ratio"] = joined["rewired_variation_ratio"] - joined["original_variation_ratio"]
    joined["delta_rashomon_capacity"] = (
        joined["rewired_rashomon_capacity"] - joined["original_rashomon_capacity"]
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    joined.to_csv(output, index=False)
    summary = joined[
        ["delta_predictive_entropy", "delta_variation_ratio", "delta_rashomon_capacity"]
    ].describe()
    summary_path = output.with_suffix(".summary.csv")
    summary.to_csv(summary_path)
    print(f"Wrote comparison table to {output}")
    print(f"Wrote comparison summary to {summary_path}")


if __name__ == "__main__":
    main()
