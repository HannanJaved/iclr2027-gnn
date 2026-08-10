from __future__ import annotations

import numpy as np
import pandas as pd


def pareto_frontier(
    frame: pd.DataFrame,
    objective_columns: list[str],
    maximize: list[bool],
) -> pd.DataFrame:
    if len(objective_columns) != len(maximize):
        raise ValueError("objective_columns and maximize must have the same length.")
    missing = [column for column in objective_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing objective columns: {missing}")

    values = frame[objective_columns].to_numpy(dtype=float)
    comparable = np.isfinite(values).all(axis=1)
    transformed = values.copy()
    for idx, should_maximize in enumerate(maximize):
        if not should_maximize:
            transformed[:, idx] = -transformed[:, idx]

    efficient = np.zeros(len(frame), dtype=bool)
    dominated_by: list[str | None] = [None] * len(frame)
    ids = frame["run_id"].astype(str).tolist() if "run_id" in frame.columns else [str(i) for i in range(len(frame))]
    for i in range(len(frame)):
        if not comparable[i]:
            continue
        candidate = transformed[i]
        dominated = False
        for j in range(len(frame)):
            if i == j or not comparable[j]:
                continue
            challenger = transformed[j]
            at_least_as_good = bool(np.all(challenger >= candidate))
            strictly_better = bool(np.any(challenger > candidate))
            if at_least_as_good and strictly_better:
                dominated = True
                dominated_by[i] = ids[j]
                break
        efficient[i] = not dominated

    out = frame.copy()
    out["is_pareto_efficient"] = efficient
    out["dominated_by"] = dominated_by
    out["pareto_rank"] = np.where(efficient, 0, 1)
    return out


def model_explanation_stability(pairwise: pd.DataFrame) -> pd.DataFrame:
    required = {"left_run_id", "right_run_id", "top_k_jaccard", "spearman", "kendall"}
    missing = required - set(pairwise.columns)
    if missing:
        raise ValueError(f"Missing pairwise explanation columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for side in ["left", "right"]:
        run_col = f"{side}_run_id"
        size_col = f"{side}_explanation_size"
        columns = {
            run_col: "run_id",
            "top_k_jaccard": "top_k_jaccard",
            "spearman": "spearman",
            "kendall": "kendall",
        }
        if size_col in pairwise.columns:
            columns[size_col] = "explanation_size"
        rows.append(pairwise[list(columns)].rename(columns=columns))

    long = pd.concat(rows, ignore_index=True)
    aggregations: dict[str, tuple[str, str]] = {
        "mean_top_k_jaccard": ("top_k_jaccard", "mean"),
        "median_top_k_jaccard": ("top_k_jaccard", "median"),
        "min_top_k_jaccard": ("top_k_jaccard", "min"),
        "mean_spearman": ("spearman", "mean"),
        "mean_kendall": ("kendall", "mean"),
        "comparison_count": ("top_k_jaccard", "size"),
    }
    if "explanation_size" in long.columns:
        aggregations["mean_explanation_size"] = ("explanation_size", "mean")
    return long.groupby("run_id", as_index=False).agg(**aggregations)


def selection_baselines(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if frame.empty:
        return pd.DataFrame()

    def add(name: str, row: pd.Series) -> None:
        rows.append(
            {
                "selection_rule": name,
                "run_id": row["run_id"],
                "validation_accuracy": row.get("validation_accuracy"),
                "test_accuracy_report_only": row.get("test_accuracy"),
                "mean_top_k_jaccard": row.get("mean_top_k_jaccard"),
                "median_top_k_jaccard": row.get("median_top_k_jaccard"),
                "is_pareto_efficient": row.get("is_pareto_efficient"),
            }
        )

    add("best_validation_accuracy", frame.sort_values("validation_accuracy", ascending=False).iloc[0])
    add("best_explanation_stability", frame.sort_values("mean_top_k_jaccard", ascending=False).iloc[0])
    if "train_loss" in frame.columns:
        median_loss = float(frame["train_loss"].median())
        idx = (frame["train_loss"] - median_loss).abs().idxmin()
        add("median_train_loss", frame.loc[idx])
    add("first_pareto_model", frame.sort_values(["is_pareto_efficient", "validation_accuracy"], ascending=[False, False]).iloc[0])
    return pd.DataFrame(rows)
