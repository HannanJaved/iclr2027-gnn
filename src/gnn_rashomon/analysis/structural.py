from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests


def spearman_table(
    frame: pd.DataFrame,
    targets: list[str],
    features: list[str],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for target in targets:
        for feature in features:
            pair = frame[[target, feature]].dropna()
            if pair.empty:
                corr = float("nan")
                p_value = float("nan")
                nobs = 0
            else:
                result = spearmanr(pair[target], pair[feature])
                corr = float(result.statistic)
                p_value = float(result.pvalue)
                nobs = int(len(pair))
            rows.append(
                {
                    "target": target,
                    "feature": feature,
                    "spearman": corr,
                    "p_value": p_value,
                    "nobs": nobs,
                }
            )
    table = pd.DataFrame(rows)
    valid = table["p_value"].notna()
    table["p_value_fdr_bh"] = np.nan
    if valid.any():
        table.loc[valid, "p_value_fdr_bh"] = multipletests(
            table.loc[valid, "p_value"].to_numpy(dtype=float),
            method="fdr_bh",
        )[1]
    return table


def _regression_formula(target: str, predictors: list[str] | None) -> str:
    terms = predictors or [
        "log_degree",
        "local_homophily",
        "neighborhood_label_entropy",
        "feature_similarity",
        "confidence",
        "correct",
    ]
    return f"{target} ~ " + " + ".join(terms)


def regression_table(
    frame: pd.DataFrame,
    target: str,
    predictors: list[str] | None = None,
) -> pd.DataFrame:
    working = frame.copy()
    working["log_degree"] = np.log1p(working["degree"])
    formula = _regression_formula(target, predictors)
    model = smf.ols(formula=formula, data=working).fit(cov_type="HC3")
    table = model.summary2().tables[1].reset_index(names="term")
    table["target"] = target
    table["r_squared"] = float(model.rsquared)
    table["nobs"] = int(model.nobs)
    p_col = "P>|z|" if "P>|z|" in table.columns else "P>|t|"
    table["p_value_fdr_bh"] = multipletests(table[p_col].to_numpy(dtype=float), method="fdr_bh")[1]
    return table


def residual_diagnostics(
    frame: pd.DataFrame,
    target: str,
    predictors: list[str] | None = None,
) -> dict[str, float | int | str]:
    working = frame.copy()
    working["log_degree"] = np.log1p(working["degree"])
    formula = _regression_formula(target, predictors)
    model = smf.ols(formula=formula, data=working).fit(cov_type="HC3")
    residuals = np.asarray(model.resid, dtype=float)
    fitted = np.asarray(model.fittedvalues, dtype=float)
    return {
        "target": target,
        "nobs": int(model.nobs),
        "r_squared": float(model.rsquared),
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals, ddof=1)) if len(residuals) > 1 else 0.0,
        "residual_min": float(np.min(residuals)),
        "residual_max": float(np.max(residuals)),
        "residual_abs_mean": float(np.mean(np.abs(residuals))),
        "fitted_min": float(np.min(fitted)),
        "fitted_max": float(np.max(fitted)),
    }
