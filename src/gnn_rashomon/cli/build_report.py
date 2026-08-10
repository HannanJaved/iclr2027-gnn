from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _dataset_from_set_id(set_id: str) -> str:
    return set_id.split("-", 1)[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate multiplicity and structural analysis outputs.")
    parser.add_argument("--set-pattern", default="*-gcn-seed-relative0.1-epochs200")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--report-name", default="seed_relative0.1")
    return parser.parse_args()


def _read_summary(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    flat: dict[str, object] = {
        "set_id": payload["set_id"],
        "dataset": payload["dataset"],
        "architecture": payload.get("architecture", "gcn"),
        "retained_count": payload["retained_count"],
        "num_nodes": payload.get("num_nodes"),
        "fraction_prediction_disagreement": payload.get("fraction_prediction_disagreement"),
    }
    for metric in ["predictive_entropy", "variation_ratio", "rashomon_capacity"]:
        values = payload.get(metric, {})
        if isinstance(values, dict):
            for key, value in values.items():
                flat[f"{metric}_{key}"] = value
    return flat


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    metrics_dir = output_dir / "metrics"
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    set_ids = sorted(path.stem.replace(".multiplicity_summary", "") for path in metrics_dir.glob(f"{args.set_pattern}.multiplicity_summary.json"))
    if not set_ids:
        raise SystemExit(f"No matching sets for pattern {args.set_pattern!r} in {metrics_dir}")

    summary_rows = [_read_summary(metrics_dir / f"{set_id}.multiplicity_summary.json") for set_id in set_ids]
    pd.DataFrame(summary_rows).to_csv(figures_dir / f"{args.report_name}.multiplicity_summary.csv", index=False)

    correlation_frames: list[pd.DataFrame] = []
    regression_frames: list[pd.DataFrame] = []
    residual_frames: list[pd.DataFrame] = []
    for set_id in set_ids:
        corr_path = metrics_dir / f"{set_id}.structure_spearman.csv"
        reg_path = metrics_dir / f"{set_id}.structure_regression.csv"
        residual_path = metrics_dir / f"{set_id}.structure_residual_diagnostics.csv"
        if corr_path.exists():
            corr = pd.read_csv(corr_path)
            corr.insert(0, "set_id", set_id)
            corr.insert(1, "dataset", _dataset_from_set_id(set_id))
            correlation_frames.append(corr)
        if reg_path.exists():
            reg = pd.read_csv(reg_path)
            reg.insert(0, "set_id", set_id)
            reg.insert(1, "dataset", _dataset_from_set_id(set_id))
            regression_frames.append(reg)
        if residual_path.exists():
            residual = pd.read_csv(residual_path)
            residual.insert(0, "set_id", set_id)
            residual.insert(1, "dataset", _dataset_from_set_id(set_id))
            residual_frames.append(residual)

    if correlation_frames:
        correlations = pd.concat(correlation_frames, ignore_index=True)
        correlations.to_csv(figures_dir / f"{args.report_name}.structure_spearman.csv", index=False)
        pivot = correlations.pivot_table(
            index=["target", "feature"],
            columns="dataset",
            values="spearman",
            aggfunc="first",
        ).reset_index()
        pivot.to_csv(figures_dir / f"{args.report_name}.structure_spearman_pivot.csv", index=False)
    if regression_frames:
        pd.concat(regression_frames, ignore_index=True).to_csv(
            figures_dir / f"{args.report_name}.structure_regression.csv",
            index=False,
        )
    if residual_frames:
        pd.concat(residual_frames, ignore_index=True).to_csv(
            figures_dir / f"{args.report_name}.structure_residual_diagnostics.csv",
            index=False,
        )

    print(f"Aggregated {len(set_ids)} sets into {figures_dir} with prefix {args.report_name}")


if __name__ == "__main__":
    main()
