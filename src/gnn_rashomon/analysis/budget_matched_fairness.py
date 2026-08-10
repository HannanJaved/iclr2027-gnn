from __future__ import annotations

import numpy as np
import pandas as pd

from gnn_rashomon.analysis.pareto import pareto_frontier


def dominates(challenger: pd.Series, candidate: pd.Series, metric: str) -> bool:
    fairness_column = f"validation_{metric}"
    values = np.asarray(
        [
            challenger["validation_accuracy"],
            challenger[fairness_column],
            candidate["validation_accuracy"],
            candidate[fairness_column],
        ],
        dtype=float,
    )
    if not np.isfinite(values).all():
        return False
    return bool(
        values[0] >= values[2]
        and values[1] <= values[3]
        and (values[0] > values[2] or values[1] < values[3])
    )


def budget_matched_comparisons(
    baseline: pd.DataFrame,
    candidate_pool: pd.DataFrame,
    *,
    metric: str,
    sample_size: int,
    draws: int,
    seed: int,
    set_id: str,
) -> pd.DataFrame:
    fairness_column = f"validation_{metric}"
    required = {"run_id", "validation_accuracy", fairness_column}
    for name, frame in (("baseline", baseline), ("candidate", candidate_pool)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} table lacks required columns: {sorted(missing)}")
    if not 1 <= sample_size <= len(candidate_pool):
        raise ValueError("sample_size must be between 1 and the candidate-pool size.")
    if draws <= 0:
        raise ValueError("draws must be positive.")

    baseline_name = (
        str(baseline["baseline"].dropna().iloc[0])
        if "baseline" in baseline and baseline["baseline"].notna().any()
        else "unspecified"
    )
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for draw in range(draws):
        indices = np.sort(rng.choice(len(candidate_pool), size=sample_size, replace=False))
        sampled = candidate_pool.iloc[indices].copy()
        frontier = pareto_frontier(
            sampled,
            objective_columns=["validation_accuracy", fairness_column],
            maximize=[True, False],
        )
        frontier = frontier[frontier["is_pareto_efficient"]].copy()
        selected = frontier.sort_values(
            ["validation_accuracy", fairness_column, "run_id"],
            ascending=[False, True, True],
            kind="mergesort",
        ).iloc[0]
        selected_accuracy_wins = []
        selected_fairness_wins = []
        selected_joint_wins = []
        frontier_joint_wins = []
        for _, baseline_row in baseline.iterrows():
            selected_accuracy_wins.append(
                float(selected["validation_accuracy"])
                > float(baseline_row["validation_accuracy"])
            )
            selected_fairness_wins.append(
                float(selected[fairness_column]) < float(baseline_row[fairness_column])
            )
            selected_joint_wins.append(dominates(selected, baseline_row, metric))
            frontier_joint_wins.append(
                any(dominates(model, baseline_row, metric) for _, model in frontier.iterrows())
            )
        rows.append(
            {
                "draw": draw,
                "set_id": set_id,
                "baseline": baseline_name,
                "fairness_metric": metric,
                "candidate_pool_size": len(candidate_pool),
                "sample_size": sample_size,
                "sampled_run_ids": ";".join(sampled["run_id"].astype(str)),
                "sampled_frontier_size": len(frontier),
                "selected_run_id": selected["run_id"],
                "selected_validation_accuracy": float(selected["validation_accuracy"]),
                "selected_validation_fairness": float(selected[fairness_column]),
                "selected_accuracy_win_rate": float(np.mean(selected_accuracy_wins)),
                "selected_fairness_win_rate": float(np.mean(selected_fairness_wins)),
                "selected_joint_win_rate": float(np.mean(selected_joint_wins)),
                "frontier_joint_win_rate": float(np.mean(frontier_joint_wins)),
            }
        )
    return pd.DataFrame(rows)


def summarize_budget_matched(draws: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "sampled_frontier_size",
        "selected_validation_accuracy",
        "selected_validation_fairness",
        "selected_accuracy_win_rate",
        "selected_fairness_win_rate",
        "selected_joint_win_rate",
        "frontier_joint_win_rate",
    ]
    rows: list[dict[str, object]] = []
    groups = draws.groupby(["set_id", "baseline", "fairness_metric"], sort=True)
    for identifiers, group in groups:
        row: dict[str, object] = dict(
            zip(["set_id", "baseline", "fairness_metric"], identifiers, strict=True)
        )
        row["draw_count"] = len(group)
        row["candidate_pool_size"] = int(group["candidate_pool_size"].iloc[0])
        row["sample_size"] = int(group["sample_size"].iloc[0])
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            low, median, high = np.quantile(values, [0.025, 0.5, 0.975])
            row[f"{metric}_mean"] = float(values.mean())
            row[f"{metric}_median"] = float(median)
            row[f"{metric}_interval_95_low"] = float(low)
            row[f"{metric}_interval_95_high"] = float(high)
        rows.append(row)
    return pd.DataFrame(rows)
