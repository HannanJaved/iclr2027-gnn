from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t as student_t


def tost_from_coefficient(
    coefficient: float,
    std_error: float,
    *,
    equivalence_margin: float,
    reference_df: float,
) -> dict[str, float | bool]:
    """Two one-sided tests for H1: |coefficient| < equivalence_margin."""
    if equivalence_margin <= 0:
        raise ValueError("equivalence_margin must be positive.")
    if std_error <= 0 or not np.isfinite(std_error):
        raise ValueError("std_error must be a positive finite value.")
    if reference_df <= 0 or not np.isfinite(reference_df):
        raise ValueError("reference_df must be positive and finite.")

    t_lower = (coefficient - (-equivalence_margin)) / std_error
    t_upper = (coefficient - equivalence_margin) / std_error
    p_lower = float(1.0 - student_t.cdf(t_lower, reference_df))
    p_upper = float(student_t.cdf(t_upper, reference_df))
    p_tost = max(p_lower, p_upper)
    critical = float(student_t.ppf(0.975, reference_df))
    ci_low = float(coefficient - critical * std_error)
    ci_high = float(coefficient + critical * std_error)
    return {
        "equivalence_margin": float(equivalence_margin),
        "coefficient": float(coefficient),
        "std_error": float(std_error),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "tost_p_lower": p_lower,
        "tost_p_upper": p_upper,
        "tost_p_value": float(p_tost),
        "equivalent_at_0_05": bool(p_tost < 0.05),
        "ci_inside_equivalence_interval": bool(
            ci_low > -equivalence_margin and ci_high < equivalence_margin
        ),
        "reference_df": float(reference_df),
    }


def equivalence_rows_from_regression(
    regression: pd.DataFrame,
    *,
    term: str = "rashomon_capacity",
    equivalence_margin: float = 0.10,
    label: str | None = None,
) -> pd.DataFrame:
    working = regression[regression["term"].astype(str) == term].copy()
    rows: list[dict[str, object]] = []
    for _, row in working.iterrows():
        if "reference_df" in row and pd.notna(row["reference_df"]):
            reference_df = float(row["reference_df"])
        elif "n" in row and pd.notna(row["n"]):
            reference_df = float(row["n"]) - 1.0
        else:
            raise ValueError("Regression row needs reference_df or n for TOST.")
        result = tost_from_coefficient(
            float(row["coefficient"]),
            float(row["std_error"]),
            equivalence_margin=equivalence_margin,
            reference_df=reference_df,
        )
        rows.append(
            {
                "label": label or str(row.get("label", "")),
                "outcome": str(row.get("outcome", "")),
                "term": term,
                "n": int(row["n"]) if "n" in row and pd.notna(row["n"]) else None,
                "cluster_count": (
                    int(row["cluster_count"])
                    if "cluster_count" in row and pd.notna(row["cluster_count"])
                    else None
                ),
                "raw_p_value": float(row["p_value"]) if "p_value" in row else float("nan"),
                "r_squared": float(row["r_squared"]) if "r_squared" in row else float("nan"),
                **result,
            }
        )
    return pd.DataFrame(rows)
