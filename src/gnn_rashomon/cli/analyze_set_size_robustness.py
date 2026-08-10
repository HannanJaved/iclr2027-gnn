from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from gnn_rashomon.analysis.set_size_robustness import (
    multiplicity_summary,
    subsample_multiplicity,
    summarize_subsamples,
)
from gnn_rashomon.rashomon.io import read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Standardize multiplicity estimates by repeatedly sampling a common number of "
            "retained models."
        )
    )
    parser.add_argument("--rashomon-set", required=True)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--output-prefix", required=True)
    return parser.parse_args()


def _resolve(path: str, project_root: Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root / candidate


def _load_probabilities(
    rashomon: dict[str, object], project_root: Path
) -> tuple[list[str], np.ndarray]:
    import torch

    retained = [str(value) for value in rashomon["retained_run_ids"]]
    runs = rashomon["runs"]
    assert isinstance(runs, dict)
    arrays = []
    for run_id in retained:
        run = runs[run_id]
        assert isinstance(run, dict)
        path = _resolve(str(run["probability_path"]), project_root)
        arrays.append(torch.load(path, map_location="cpu", weights_only=True).numpy())
    return retained, np.stack(arrays)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[3]
    rashomon_path = Path(args.rashomon_set)
    rashomon = read_json(rashomon_path)
    model_ids, probabilities = _load_probabilities(rashomon, project_root)
    draws = subsample_multiplicity(
        probabilities,
        model_ids,
        sample_size=args.sample_size,
        draws=args.draws,
        seed=args.seed,
    )

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    draw_path = Path(f"{prefix}.draws.csv")
    summary_path = Path(f"{prefix}.summary.json")
    draws.to_csv(draw_path, index=False)
    summary = {
        "dataset": rashomon["dataset"],
        "architecture": rashomon["architecture"],
        "set_id": rashomon["set_id"],
        "rashomon_set": str(rashomon_path),
        "retained_model_count": len(model_ids),
        "sample_size": args.sample_size,
        "draw_count": args.draws,
        "seed": args.seed,
        "full_set_metrics": multiplicity_summary(probabilities),
        "subsample_metrics": summarize_subsamples(draws),
        "draw_table": str(draw_path),
        "probability_diameter_definition": "maximum pairwise total-variation distance",
        "pairwise_robustness_metrics": ["mean", "node-level 0.95 quantile"],
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
