from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy.stats import binomtest, ttest_1samp, wilcoxon
from statsmodels.stats.multitest import multipletests

GROUP_COLUMNS = ["dataset", "mode", "strength", "target_homophily", "tau"]
EFFECT_COLUMNS = [
    "delta_mean_probability_diameter",
    "delta_disagreement_fraction",
    "matched_treatment_control_delta_diameter",
]


def bootstrap_mean_interval(
    values: np.ndarray,
    *,
    seed: int,
    samples: int = 10_000,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def summarize_realization_effects(
    realizations: pd.DataFrame,
    *,
    seed: int = 20260715,
    bootstrap_samples: int = 10_000,
) -> pd.DataFrame:
    missing = set(GROUP_COLUMNS + EFFECT_COLUMNS) - set(realizations.columns)
    if missing:
        raise ValueError(f"Missing repeated-rewiring columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    grouped = realizations.groupby(GROUP_COLUMNS, dropna=False, sort=True)
    for group_index, (group_values, group) in enumerate(grouped):
        identifiers = dict(zip(GROUP_COLUMNS, group_values, strict=True))
        for effect_index, effect in enumerate(EFFECT_COLUMNS):
            values = group[effect].dropna().to_numpy(dtype=float)
            if values.size == 0:
                continue
            t_result = ttest_1samp(values, popmean=0.0)
            try:
                signed_rank_p = float(wilcoxon(values, alternative="two-sided").pvalue)
            except ValueError:
                signed_rank_p = 1.0
            positive_count = int((values > 0).sum())
            negative_count = int((values < 0).sum())
            zero_count = int((values == 0).sum())
            nonzero_count = positive_count + negative_count
            sign_p = (
                float(binomtest(positive_count, nonzero_count, 0.5).pvalue)
                if nonzero_count
                else 1.0
            )
            directional_sign_p = (
                float(
                    binomtest(
                        positive_count, nonzero_count, 0.5, alternative="greater"
                    ).pvalue
                )
                if nonzero_count
                else 1.0
            )
            low, high = bootstrap_mean_interval(
                values,
                seed=seed + group_index * len(EFFECT_COLUMNS) + effect_index,
                samples=bootstrap_samples,
            )
            rows.append(
                {
                    **identifiers,
                    "effect": effect,
                    "realization_count": int(values.size),
                    "mean_effect": float(values.mean()),
                    "standard_deviation": float(values.std(ddof=1))
                    if values.size > 1
                    else float("nan"),
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "t_statistic": float(t_result.statistic),
                    "t_reference_df": int(values.size - 1),
                    "t_p_value": float(t_result.pvalue),
                    "signed_rank_p_value": signed_rank_p,
                    "positive_effect_count": positive_count,
                    "negative_effect_count": negative_count,
                    "zero_effect_count": zero_count,
                    "sign_test_p_value": sign_p,
                    "directional_sign_p_value": directional_sign_p,
                    "inference_unit": "independently seeded rewired graph realization",
                }
            )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    for raw_column, prefix in [
        ("t_p_value", "t"),
        ("signed_rank_p_value", "signed_rank"),
        ("sign_test_p_value", "sign_test"),
        ("directional_sign_p_value", "directional_sign"),
    ]:
        summary[f"{prefix}_bh_p_value"] = np.nan
        summary[f"{prefix}_bonferroni_p_value"] = np.nan
        for _, family in summary.groupby("effect"):
            finite = family[raw_column].notna()
            indices = family.index[finite]
            raw = family.loc[indices, raw_column].to_numpy(dtype=float)
            if raw.size == 0:
                continue
            summary.loc[indices, f"{prefix}_bh_p_value"] = multipletests(
                raw, method="fdr_bh"
            )[1]
            summary.loc[indices, f"{prefix}_bonferroni_p_value"] = multipletests(
                raw, method="bonferroni"
            )[1]
    return summary


def realization_frame(payloads: Iterable[dict[str, object]]) -> pd.DataFrame:
    rows = list(payloads)
    if not rows:
        raise ValueError("At least one repeated-rewiring realization is required.")
    return pd.DataFrame(rows)
