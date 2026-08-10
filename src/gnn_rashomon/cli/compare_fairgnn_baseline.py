from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

FAIRNESS_METRICS = [
    "demographic_parity_gap",
    "equalized_odds_gap",
    "degree_disparity",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare fairness-baseline runs with Rashomon fairness Pareto frontiers."
    )
    parser.add_argument("--baseline-table", required=True)
    parser.add_argument("--pareto-models", nargs="+", required=True)
    parser.add_argument("--output-prefix", default="outputs/figures/fairgnn_baseline_comparison")
    return parser.parse_args()


def _infer_set_id(path: Path, frame: pd.DataFrame) -> str:
    if "set_id" in frame.columns and frame["set_id"].notna().any():
        return str(frame["set_id"].dropna().iloc[0])
    name = path.name
    return name.split(".fairness_")[0] if ".fairness_" in name else path.stem


def _infer_metric(path: Path, frame: pd.DataFrame) -> str:
    for metric in FAIRNESS_METRICS:
        if metric in path.name and f"validation_{metric}" in frame.columns:
            return metric
    candidates = [metric for metric in FAIRNESS_METRICS if f"validation_{metric}" in frame.columns]
    if len(candidates) == 1:
        return candidates[0]
    if candidates:
        for metric in candidates:
            if f"fairness_{metric}" in path.name:
                return metric
    raise ValueError(f"Could not infer fairness metric for {path}.")


def _dominates(challenger: pd.Series, candidate: pd.Series, metric: str) -> bool:
    selection_metric = f"validation_{metric}"
    challenger_val = float(challenger["validation_accuracy"])
    challenger_fair = float(challenger[selection_metric])
    candidate_val = float(candidate["validation_accuracy"])
    candidate_fair = float(candidate[selection_metric])
    if not np.isfinite([challenger_val, challenger_fair, candidate_val, candidate_fair]).all():
        return False
    at_least_as_good = challenger_val >= candidate_val and challenger_fair <= candidate_fair
    strictly_better = challenger_val > candidate_val or challenger_fair < candidate_fair
    return bool(at_least_as_good and strictly_better)


def _comparison_rows(
    baseline: pd.DataFrame,
    pareto_models: pd.DataFrame,
    set_id: str,
    metric: str,
) -> list[dict[str, object]]:
    selection_metric = f"validation_{metric}"
    report_metric = f"test_{metric}_report_only"
    required = {
        "run_id",
        "validation_accuracy",
        "test_accuracy_report_only",
        selection_metric,
        report_metric,
    }
    for name, frame in (("baseline", baseline), ("Pareto", pareto_models)):
        missing = required.difference(frame.columns)
        if missing:
            raise ValueError(f"{name} table lacks split-specific columns: {sorted(missing)}")
    frontier = pareto_models[pareto_models["is_pareto_efficient"].astype(bool)].copy()
    if frontier.empty:
        raise ValueError(f"No Pareto-efficient models found for {set_id} and {metric}.")
    selected = frontier.sort_values(
        ["validation_accuracy", selection_metric, "run_id"],
        ascending=[False, True, True],
        kind="mergesort",
    ).iloc[0]
    rows: list[dict[str, object]] = []
    baseline_name = (
        str(baseline["baseline"].dropna().iloc[0])
        if "baseline" in baseline.columns and baseline["baseline"].notna().any()
        else "unspecified"
    )
    for _, baseline_row in baseline.iterrows():
        dominating = [
            str(row["run_id"])
            for _, row in frontier.iterrows()
            if _dominates(row, baseline_row, metric)
        ]
        rows.append(
            {
                "set_id": set_id,
                "fairness_metric": metric,
                "baseline": baseline_name,
                "baseline_run_id": baseline_row["run_id"],
                "baseline_validation_accuracy": baseline_row["validation_accuracy"],
                "baseline_test_accuracy_report_only": baseline_row["test_accuracy_report_only"],
                "baseline_validation_metric": baseline_row[selection_metric],
                "baseline_test_metric_report_only": baseline_row[report_metric],
                "pareto_model_count": int(len(frontier)),
                "rashomon_frontier_best_validation_accuracy": float(
                    frontier["validation_accuracy"].max()
                ),
                "rashomon_frontier_best_validation_metric": float(frontier[selection_metric].min()),
                "rashomon_frontier_dominates_baseline": bool(dominating),
                "dominating_pareto_run_ids": ";".join(dominating),
                "selected_pareto_run_id": selected["run_id"],
                "selected_validation_accuracy": selected["validation_accuracy"],
                "selected_validation_metric": selected[selection_metric],
                "selected_test_metric_report_only": selected[report_metric],
                "selected_accuracy_beats_baseline": bool(
                    float(selected["validation_accuracy"])
                    > float(baseline_row["validation_accuracy"])
                ),
                "selected_fairness_beats_baseline": bool(
                    float(selected[selection_metric]) < float(baseline_row[selection_metric])
                ),
                "selected_jointly_dominates_baseline": _dominates(selected, baseline_row, metric),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    baseline = pd.read_csv(args.baseline_table)
    rows: list[dict[str, object]] = []
    for raw_path in args.pareto_models:
        path = Path(raw_path)
        models = pd.read_csv(path)
        metric = _infer_metric(path, models)
        set_id = _infer_set_id(path, models)
        rows.extend(_comparison_rows(baseline, models, set_id, metric))

    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    table_path = Path(f"{output_prefix}.csv")
    summary_path = Path(f"{output_prefix}.summary.json")
    table = pd.DataFrame(rows)
    table.to_csv(table_path, index=False)
    grouped = {}
    selected_joint_grouped = {}
    if len(table):
        grouped = {
            f"{set_id}|{metric}": float(value)
            for (set_id, metric), value in table.groupby(["set_id", "fairness_metric"])[
                "rashomon_frontier_dominates_baseline"
            ]
            .mean()
            .items()
        }
        selected_joint_grouped = {
            f"{set_id}|{metric}": float(value)
            for (set_id, metric), value in table.groupby(["set_id", "fairness_metric"])[
                "selected_jointly_dominates_baseline"
            ]
            .mean()
            .items()
        }
    summary = {
        "baseline": str(baseline["baseline"].dropna().iloc[0])
        if "baseline" in baseline.columns and baseline["baseline"].notna().any()
        else "unspecified",
        "baseline_table": str(args.baseline_table),
        "pareto_model_tables": list(args.pareto_models),
        "comparison_table": str(table_path),
        "rows": int(len(table)),
        "selection_uses_test_metrics": False,
        "test_accuracy_is_report_only": True,
        "dominance_rate": float(table["rashomon_frontier_dominates_baseline"].mean())
        if len(table)
        else float("nan"),
        "by_set_and_metric": grouped,
        "selected_joint_dominance_rate": float(table["selected_jointly_dominates_baseline"].mean())
        if len(table)
        else float("nan"),
        "selected_accuracy_win_rate": float(table["selected_accuracy_beats_baseline"].mean())
        if len(table)
        else float("nan"),
        "selected_fairness_win_rate": float(table["selected_fairness_beats_baseline"].mean())
        if len(table)
        else float("nan"),
        "selected_joint_by_set_and_metric": selected_joint_grouped,
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote fairness-baseline comparison table to {table_path}")
    print(f"Wrote fairness-baseline comparison summary to {summary_path}")


if __name__ == "__main__":
    main()
