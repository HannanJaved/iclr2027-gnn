from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from gnn_rashomon.analysis.performance_sensitivity import constrain_by_validation_accuracy
from gnn_rashomon.multiplicity.rashomon_capacity import probability_diameter
from gnn_rashomon.multiplicity.variation_ratio import variation_ratio
from gnn_rashomon.rashomon.io import read_json, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build validation-performance-constrained Rashomon sensitivity sets."
    )
    parser.add_argument("--rashomon-set", action="append", required=True)
    parser.add_argument(
        "--validation-accuracy-delta",
        action="append",
        type=float,
        required=True,
        help="Absolute validation-accuracy tolerance, such as 0.02 or 0.05.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/performance_sensitivity",
    )
    parser.add_argument(
        "--capacity-method",
        choices=["exact", "chunked", "approximate"],
        default="chunked",
    )
    parser.add_argument("--capacity-chunk-size", type=int, default=16)
    parser.add_argument("--capacity-sample-pairs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20260715)
    return parser.parse_args()


def _load_probability(path: str) -> np.ndarray:
    import torch

    return torch.load(path, map_location="cpu", weights_only=True).detach().cpu().numpy()


def _multiplicity_summary(payload: dict[str, object], args: argparse.Namespace) -> dict[str, float]:
    retained = [str(run_id) for run_id in payload["retained_run_ids"]]
    runs = payload["runs"]
    probabilities = np.stack(
        [_load_probability(str(runs[run_id]["probability_path"])) for run_id in retained]
    )
    predictions = probabilities.argmax(axis=-1)
    disagreement = (predictions != predictions[0:1]).any(axis=0)
    diameter = probability_diameter(
        probabilities,
        method=args.capacity_method,
        chunk_size=args.capacity_chunk_size,
        sample_pairs=args.capacity_sample_pairs,
        seed=args.seed,
    )
    return {
        "fraction_prediction_disagreement": float(disagreement.mean()),
        "mean_probability_diameter": float(np.mean(diameter)),
        "mean_variation_ratio": float(np.mean(variation_ratio(probabilities))),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    set_dir = output_dir / "rashomon_sets"
    rows: list[dict[str, object]] = []
    for raw_path in args.rashomon_set:
        source_path = Path(raw_path)
        source = read_json(source_path)
        for delta in sorted(set(args.validation_accuracy_delta)):
            payload, summary = constrain_by_validation_accuracy(source, delta)
            summary.update(_multiplicity_summary(payload, args))
            summary["source_path"] = str(source_path)
            set_path = set_dir / f"{payload['set_id']}.json"
            write_json(set_path, payload)
            summary["derived_set_path"] = str(set_path)
            rows.append(summary)
            print(
                f"{payload['set_id']}: retained={payload['retained_count']} "
                f"disagreement={summary['fraction_prediction_disagreement']:.4f}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "performance_constrained_rashomon_summary.csv"
    json_path = output_dir / "performance_constrained_rashomon_summary.json"
    frame = pd.DataFrame(rows)
    if csv_path.exists():
        frame = pd.concat([pd.read_csv(csv_path), frame], ignore_index=True)
        frame = frame.drop_duplicates(
            subset=["parent_set_id", "validation_accuracy_delta"], keep="last"
        )
    frame = frame.sort_values(["dataset", "parent_set_id", "validation_accuracy_delta"])
    frame.to_csv(csv_path, index=False)
    json_path.write_text(
        json.dumps(frame.to_dict(orient="records"), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
