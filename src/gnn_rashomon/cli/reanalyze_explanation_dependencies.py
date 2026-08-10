from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from gnn_rashomon.explainability.comparison import instability_regression, summarize_instability


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild explanation node summaries and dependency-aware regressions."
    )
    parser.add_argument("--metadata", action="append", required=True)
    return parser.parse_args()


def reanalyze(metadata_path: Path) -> tuple[Path, Path]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    outputs = metadata["outputs"]
    selected_nodes = pd.read_csv(outputs["selected_nodes"])
    pairwise = pd.read_csv(outputs["pairwise"])
    summary = summarize_instability(pairwise, selected_nodes)
    regression = instability_regression(summary)

    summary_path = Path(outputs["node_summary"])
    regression_path = Path(outputs["regression"])
    summary.to_csv(summary_path, index=False)
    regression.to_csv(regression_path, index=False)
    return summary_path, regression_path


def main() -> None:
    args = parse_args()
    for raw_path in args.metadata:
        summary_path, regression_path = reanalyze(Path(raw_path))
        print(f"Rebuilt node summary: {summary_path}")
        print(f"Rebuilt dependency-aware regression: {regression_path}")


if __name__ == "__main__":
    main()
