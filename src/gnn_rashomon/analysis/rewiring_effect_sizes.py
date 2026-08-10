from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

PRACTICAL_THRESHOLDS = (0.01, 0.05, 0.10)


def relative_increase(rewired: float, original: float) -> float:
    if not np.isfinite(rewired) or not np.isfinite(original):
        return float("nan")
    if abs(original) < 1e-12:
        return float("nan")
    return float((rewired - original) / original)


def enrich_realization_effect_sizes(realizations: pd.DataFrame) -> pd.DataFrame:
    frame = realizations.copy()
    if {
        "rewired_mean_probability_diameter",
        "original_mean_probability_diameter",
    }.issubset(frame.columns):
        frame["relative_delta_mean_probability_diameter"] = [
            relative_increase(float(rewired), float(original))
            for rewired, original in zip(
                frame["rewired_mean_probability_diameter"],
                frame["original_mean_probability_diameter"],
                strict=True,
            )
        ]
    elif {
        "delta_mean_probability_diameter",
        "original_mean_probability_diameter",
    }.issubset(frame.columns):
        frame["relative_delta_mean_probability_diameter"] = (
            frame["delta_mean_probability_diameter"]
            / frame["original_mean_probability_diameter"].replace(0, np.nan)
        )
    if {
        "rewired_disagreement_fraction",
        "original_disagreement_fraction",
    }.issubset(frame.columns):
        frame["relative_delta_disagreement_fraction"] = [
            relative_increase(float(rewired), float(original))
            for rewired, original in zip(
                frame["rewired_disagreement_fraction"],
                frame["original_disagreement_fraction"],
                strict=True,
            )
        ]
    if (
        "matched_treatment_control_delta_diameter" in frame.columns
        and "delta_treatment_entropy_mean" in frame.columns
    ):
        denom = frame["delta_treatment_entropy_mean"].replace(0, np.nan)
        frame["standardized_matched_delta_per_entropy"] = (
            frame["matched_treatment_control_delta_diameter"] / denom
        )
    return frame


def summarize_effect_sizes(realizations: pd.DataFrame) -> pd.DataFrame:
    frame = enrich_realization_effect_sizes(realizations)
    valid = frame
    if "valid_for_inference" in frame.columns:
        valid = frame.loc[frame["valid_for_inference"].astype(bool)].copy()
    group_cols = [
        column
        for column in ["dataset", "mode", "strength", "target_homophily", "tau"]
        if column in valid.columns
    ]
    rows: list[dict[str, object]] = []
    for keys, group in valid.groupby(group_cols, dropna=False, sort=True):
        identifiers = dict(zip(group_cols, keys if isinstance(keys, tuple) else (keys,), strict=True))
        delta = group["delta_mean_probability_diameter"].dropna().to_numpy(dtype=float)
        relative = group["relative_delta_mean_probability_diameter"].dropna().to_numpy(dtype=float)
        row: dict[str, object] = {
            **identifiers,
            "realization_count": int(len(group)),
            "mean_absolute_delta_D": float(delta.mean()) if delta.size else float("nan"),
            "median_absolute_delta_D": float(np.median(delta)) if delta.size else float("nan"),
            "mean_relative_delta_D": float(relative.mean()) if relative.size else float("nan"),
            "median_relative_delta_D": float(np.median(relative)) if relative.size else float("nan"),
            "fraction_positive_delta_D": float((delta > 0).mean()) if delta.size else float("nan"),
        }
        if "standardized_matched_delta_per_entropy" in group.columns:
            std_vals = group["standardized_matched_delta_per_entropy"].dropna().to_numpy(dtype=float)
            row["mean_standardized_matched_delta_per_entropy"] = (
                float(std_vals.mean()) if std_vals.size else float("nan")
            )
        rows.append(row)
    return pd.DataFrame(rows)


def dose_response_table(realizations: pd.DataFrame) -> pd.DataFrame:
    frame = enrich_realization_effect_sizes(realizations)
    if "valid_for_inference" in frame.columns:
        frame = frame.loc[frame["valid_for_inference"].astype(bool)].copy()
    frame = frame[frame["mode"].astype(str) == "random"].copy()
    if frame.empty:
        return frame
    grouped = (
        frame.groupby(["dataset", "strength"], dropna=False, sort=True)
        .agg(
            realization_count=("delta_mean_probability_diameter", "size"),
            mean_delta_D=("delta_mean_probability_diameter", "mean"),
            mean_relative_delta_D=("relative_delta_mean_probability_diameter", "mean"),
            mean_delta_disagreement=("delta_disagreement_fraction", "mean"),
            mean_original_D=("original_mean_probability_diameter", "mean"),
            mean_rewired_D=("rewired_mean_probability_diameter", "mean"),
        )
        .reset_index()
    )
    return grouped.sort_values(["dataset", "strength"])


def summarize_node_delta_distribution(node_table: pd.DataFrame) -> dict[str, float]:
    if "delta_probability_diameter" not in node_table.columns:
        raise ValueError("Expected delta_probability_diameter column.")
    values = node_table["delta_probability_diameter"].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("No finite node-level delta diameters.")
    quantiles = np.quantile(finite, [0.05, 0.25, 0.5, 0.75, 0.95])
    summary = {
        "n_nodes": int(finite.size),
        "mean_delta_D_i": float(finite.mean()),
        "std_delta_D_i": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        "q05_delta_D_i": float(quantiles[0]),
        "q25_delta_D_i": float(quantiles[1]),
        "q50_delta_D_i": float(quantiles[2]),
        "q75_delta_D_i": float(quantiles[3]),
        "q95_delta_D_i": float(quantiles[4]),
        "fraction_delta_D_i_positive": float((finite > 0).mean()),
    }
    for threshold in PRACTICAL_THRESHOLDS:
        summary[f"fraction_delta_D_i_gt_{str(threshold).replace('.', 'p')}"] = float(
            (finite > threshold).mean()
        )
    return summary


def summarize_node_delta_paths(paths: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in paths:
        table = pd.read_csv(path)
        summary = summarize_node_delta_distribution(table)
        summary["source_path"] = str(path)
        summary["stem"] = path.stem
        rows.append(summary)
    return pd.DataFrame(rows)
