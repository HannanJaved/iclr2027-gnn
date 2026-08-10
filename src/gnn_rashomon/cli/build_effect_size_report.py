from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add practical effect-size context to existing regression tables."
    )
    parser.add_argument("--summary", action="append", required=True, help="Middle-link summary JSON.")
    parser.add_argument("--regression-csv", action="append", required=True, help="Regression CSV.")
    parser.add_argument(
        "--outcome",
        default="capacity_with_middle_link",
        help="Regression outcome label to summarize.",
    )
    parser.add_argument(
        "--output-csv",
        default="outputs/figures/middle_link_effect_sizes.csv",
    )
    parser.add_argument(
        "--output-json",
        default="outputs/figures/middle_link_effect_sizes.summary.json",
    )
    return parser.parse_args()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_json(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return _project_root() / path


def _summary_entries(summary: dict[str, object]) -> list[dict[str, object]]:
    entries = summary.get("datasets", summary.get("sets", []))
    out: list[dict[str, object]] = []
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        label = str(entry.get("label", entry.get("dataset", entry.get("set_id", ""))))
        out.append(
            {
                "label": label,
                "dataset": str(entry.get("dataset", label)),
                "set_id": str(entry.get("set_id", "")),
                "middle_link_nodes_csv": str(entry["middle_link_nodes_csv"]),
            }
        )
    return out


def _match_entry(row: pd.Series, entries: list[dict[str, object]]) -> dict[str, object] | None:
    candidates = [
        str(row.get("label", "")),
        str(row.get("dataset", "")),
        str(row.get("set_id", "")),
    ]
    for entry in entries:
        labels = {str(entry.get("label", "")), str(entry.get("dataset", "")), str(entry.get("set_id", ""))}
        if any(candidate and candidate in labels for candidate in candidates):
            return entry
    return None


def _partial_r_squared(frame: pd.DataFrame, formula: str, term: str) -> float:
    try:
        import statsmodels.formula.api as smf
    except Exception:
        return float("nan")
    if "~" not in formula:
        return float("nan")
    lhs, rhs = formula.split("~", maxsplit=1)
    terms = [part.strip() for part in rhs.split("+")]
    if term not in terms:
        return float("nan")
    reduced_terms = [part for part in terms if part != term]
    if not reduced_terms:
        return float("nan")
    full = smf.ols(formula, data=frame).fit()
    reduced = smf.ols(f"{lhs.strip()} ~ {' + '.join(reduced_terms)}", data=frame).fit()
    denom = 1.0 - float(reduced.rsquared)
    if denom <= 0:
        return float("nan")
    return float((float(full.rsquared) - float(reduced.rsquared)) / denom)


def _effect_rows(
    regression: pd.DataFrame,
    entries: list[dict[str, object]],
    outcome: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    data_cache: dict[str, pd.DataFrame] = {}
    working = regression[regression["outcome"].astype(str) == outcome].copy()
    for _, row_series in working.iterrows():
        term = str(row_series.get("term", ""))
        if term == "Intercept":
            continue
        entry = _match_entry(row_series, entries)
        if entry is None:
            continue
        node_csv = _resolve(str(entry["middle_link_nodes_csv"]))
        if str(node_csv) not in data_cache:
            data_cache[str(node_csv)] = pd.read_csv(node_csv)
        frame = data_cache[str(node_csv)]
        formula = str(row_series.get("formula", ""))
        if "~" not in formula:
            continue
        outcome_var = formula.split("~", maxsplit=1)[0].strip()
        if term not in frame.columns or outcome_var not in frame.columns:
            continue
        pair = frame[[outcome_var, term]].replace([np.inf, -np.inf], np.nan).dropna()
        if len(pair) < 3:
            continue
        coef = float(row_series.get("Coef.", row_series.get("coefficient", np.nan)))
        x_sd = float(pair[term].std(ddof=1))
        y_sd = float(pair[outcome_var].std(ddof=1))
        one_sd_delta = float(coef * x_sd)
        standardized_beta = float(one_sd_delta / y_sd) if y_sd > 0 else float("nan")
        rows.append(
            {
                "label": entry["label"],
                "dataset": entry["dataset"],
                "set_id": entry["set_id"],
                "outcome": outcome,
                "outcome_variable": outcome_var,
                "term": term,
                "coefficient": coef,
                "term_sd": x_sd,
                "outcome_sd": y_sd,
                "one_sd_term_delta_outcome": one_sd_delta,
                "standardized_beta": standardized_beta,
                "partial_r_squared": _partial_r_squared(frame, formula, term),
                "model_r_squared": float(row_series.get("r_squared", np.nan)),
                "n": int(row_series.get("nobs", len(pair))),
                "p_value": float(row_series.get("raw_p", row_series.get("P>|z|", np.nan))),
                "bh_q": float(row_series.get("bh_q_within_outcome", np.nan)),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    entries: list[dict[str, object]] = []
    for summary_path in args.summary:
        entries.extend(_summary_entries(_read_json(summary_path)))

    all_rows: list[dict[str, object]] = []
    for regression_path in args.regression_csv:
        regression = pd.read_csv(regression_path)
        all_rows.extend(_effect_rows(regression, entries, outcome=args.outcome))

    table = pd.DataFrame(all_rows)
    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_csv, index=False)

    focal = table[table["term"] == "probability_variance_mean"] if not table.empty else table
    summary = {
        "output_csv": str(output_csv),
        "rows": int(len(table)),
        "focal_rows": int(len(focal)),
        "outcome": args.outcome,
        "note": (
            "standardized_beta is coef * SD(term) / SD(outcome); "
            "one_sd_term_delta_outcome is the raw outcome-scale change for a one-SD term increase."
        ),
    }
    if not focal.empty:
        summary["focal_standardized_beta_range"] = [
            float(focal["standardized_beta"].min()),
            float(focal["standardized_beta"].max()),
        ]
        summary["focal_partial_r_squared_range"] = [
            float(focal["partial_r_squared"].min()),
            float(focal["partial_r_squared"].max()),
        ]
    output_json.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote effect-size table to {output_csv}")
    print(f"Wrote effect-size summary to {output_json}")


if __name__ == "__main__":
    main()
