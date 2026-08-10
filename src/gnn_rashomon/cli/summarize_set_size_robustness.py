from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from gnn_rashomon.rashomon.io import read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine set-size robustness JSON summaries into one flat table."
    )
    parser.add_argument("--summary-glob", action="append", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def flatten_summary(payload: dict[str, object]) -> dict[str, object]:
    row: dict[str, object] = {
        "dataset": payload["dataset"],
        "architecture": payload["architecture"],
        "set_id": payload["set_id"],
        "retained_model_count": payload["retained_model_count"],
        "sample_size": payload["sample_size"],
        "draw_count": payload["draw_count"],
    }
    full_set = payload["full_set_metrics"]
    subsamples = payload["subsample_metrics"]
    assert isinstance(full_set, dict)
    assert isinstance(subsamples, dict)
    for metric, value in full_set.items():
        row[f"full_set_{metric}"] = value
    for metric, estimates in subsamples.items():
        assert isinstance(estimates, dict)
        for statistic, value in estimates.items():
            row[f"subsample_{metric}_{statistic}"] = value
    return row


def main() -> None:
    args = parse_args()
    paths: list[Path] = []
    for pattern in args.summary_glob:
        paths.extend(sorted(Path().glob(pattern)))
    paths = sorted(set(paths))
    if not paths:
        raise SystemExit("No set-size robustness summaries were found.")
    table = pd.DataFrame(flatten_summary(read_json(path)) for path in paths)
    table = table.sort_values(["dataset", "architecture", "set_id"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(f"Wrote {len(table)} set-size robustness summaries to {output}")


if __name__ == "__main__":
    main()
