from __future__ import annotations

import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

DELTA_METRICS = [
    "delta_neighborhood_label_entropy",
    "delta_local_homophily",
    "delta_rashomon_capacity",
    "delta_predictive_entropy",
    "delta_variation_ratio",
    "delta_confidence",
    "delta_correct",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare treatment and control nodes for local-entanglement rewiring."
    )
    parser.add_argument("--rewiring-metadata", required=True)
    parser.add_argument("--original-structure", required=True)
    parser.add_argument("--rewired-structure", required=True)
    parser.add_argument("--delta-table", required=True)
    parser.add_argument(
        "--output-prefix",
        default="outputs/figures/cora_local_entanglement_treatment_control",
        help="Prefix for node, summary, test, regression, and plot outputs.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _read_metadata(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _prefix_except_node_id(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    renamed = {column: f"{prefix}_{column}" for column in frame.columns if column != "node_id"}
    return frame.rename(columns=renamed)


def _bootstrap_ci(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return float("nan"), float("nan")
    if samples <= 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(clean, size=(samples, clean.size), replace=True)
    means = draws.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _safe_std(values: np.ndarray) -> float:
    clean = values[np.isfinite(values)]
    if clean.size < 2:
        return float("nan")
    return float(np.std(clean, ddof=1))


def _safe_mean(values: np.ndarray) -> float:
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return float("nan")
    return float(np.mean(clean))


def _safe_median(values: np.ndarray) -> float:
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return float("nan")
    return float(np.median(clean))


def build_treatment_control_table(
    metadata: dict[str, object],
    original_structure: pd.DataFrame,
    rewired_structure: pd.DataFrame,
    delta_table: pd.DataFrame,
) -> pd.DataFrame:
    treatment_nodes = [int(node) for node in metadata.get("treatment_nodes", [])]
    control_nodes = [int(node) for node in metadata.get("control_nodes", [])]
    if not treatment_nodes:
        raise ValueError("Rewiring metadata does not contain treatment_nodes.")
    if not control_nodes:
        raise ValueError("Rewiring metadata does not contain control_nodes.")
    if len(treatment_nodes) != len(control_nodes):
        raise ValueError("treatment_nodes and control_nodes must have the same length.")

    pairs = pd.DataFrame(
        {
            "node_id": treatment_nodes + control_nodes,
            "group": ["treatment"] * len(treatment_nodes) + ["control"] * len(control_nodes),
            "pair_id": list(range(len(treatment_nodes))) + list(range(len(control_nodes))),
            "matched_node_id": control_nodes + treatment_nodes,
        }
    )

    original = _prefix_except_node_id(original_structure, "original")
    rewired = _prefix_except_node_id(rewired_structure, "rewired")
    delta = delta_table.rename(columns={"original_node_id": "node_id"}).drop(
        columns=["rewired_node_id"], errors="ignore"
    )
    table = pairs.merge(original, on="node_id", how="left", validate="one_to_one")
    table = table.merge(rewired, on="node_id", how="left", validate="one_to_one")
    table = table.merge(delta, on="node_id", how="left", validate="one_to_one")

    required = [
        "original_neighborhood_label_entropy",
        "rewired_neighborhood_label_entropy",
        "original_local_homophily",
        "rewired_local_homophily",
        "original_confidence",
        "rewired_confidence",
        "original_correct",
        "rewired_correct",
        "delta_rashomon_capacity",
        "delta_predictive_entropy",
        "delta_variation_ratio",
    ]
    missing = [column for column in required if column not in table.columns]
    if missing:
        raise ValueError(f"Missing required columns after join: {missing}")
    if table[required].isna().any().any():
        missing_nodes = table.loc[table[required].isna().any(axis=1), "node_id"].tolist()
        raise ValueError(f"Missing treatment/control rows for node ids: {missing_nodes[:20]}")

    table["delta_neighborhood_label_entropy"] = (
        table["rewired_neighborhood_label_entropy"] - table["original_neighborhood_label_entropy"]
    )
    table["delta_local_homophily"] = (
        table["rewired_local_homophily"] - table["original_local_homophily"]
    )
    table["delta_confidence"] = table["rewired_confidence"] - table["original_confidence"]
    table["delta_correct"] = table["rewired_correct"] - table["original_correct"]
    table["treatment_indicator"] = (table["group"] == "treatment").astype(int)
    return table.sort_values(["pair_id", "group"], ascending=[True, False]).reset_index(drop=True)


def _paired_effects(table: pd.DataFrame, metric: str) -> np.ndarray:
    pivot = table.pivot(index="pair_id", columns="group", values=metric)
    pivot = pivot.dropna(subset=["treatment", "control"])
    return (pivot["treatment"] - pivot["control"]).to_numpy(dtype=float)


def summarize_groups(
    table: pd.DataFrame,
    metrics: Iterable[str] = DELTA_METRICS,
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric_pos, metric in enumerate(metrics):
        for group_pos, (group, group_table) in enumerate(table.groupby("group", sort=True)):
            values = group_table[metric].to_numpy(dtype=float)
            low, high = _bootstrap_ci(
                values, bootstrap_samples, seed + 100 * metric_pos + group_pos
            )
            rows.append(
                {
                    "metric": metric,
                    "group": group,
                    "n": int(np.isfinite(values).sum()),
                    "mean": _safe_mean(values),
                    "median": _safe_median(values),
                    "std": _safe_std(values),
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                }
            )
    return pd.DataFrame(rows)


def run_tests(
    table: pd.DataFrame,
    metrics: Iterable[str] = DELTA_METRICS,
    bootstrap_samples: int = 2000,
    seed: int = 0,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    try:
        from scipy.stats import wilcoxon
    except Exception:
        wilcoxon = None

    for metric_pos, metric in enumerate(metrics):
        grouped = {
            group: group_table[metric].to_numpy(dtype=float)
            for group, group_table in table.groupby("group", sort=True)
        }
        for group, values in grouped.items():
            finite = values[np.isfinite(values)]
            statistic = float("nan")
            p_value = float("nan")
            if wilcoxon is not None and finite.size:
                try:
                    result = wilcoxon(finite, zero_method="zsplit")
                    statistic = float(result.statistic)
                    p_value = float(result.pvalue)
                except ValueError:
                    pass
            rows.append(
                {
                    "metric": metric,
                    "comparison": f"{group}_vs_zero",
                    "n": int(finite.size),
                    "effect_mean": _safe_mean(finite),
                    "effect_median": _safe_median(finite),
                    "bootstrap_ci_low": _bootstrap_ci(finite, bootstrap_samples, seed + metric_pos)[
                        0
                    ],
                    "bootstrap_ci_high": _bootstrap_ci(
                        finite, bootstrap_samples, seed + metric_pos
                    )[1],
                    "test": "wilcoxon_signed_rank",
                    "statistic": statistic,
                    "p_value": p_value,
                }
            )

        paired = _paired_effects(table, metric)
        finite_paired = paired[np.isfinite(paired)]
        low, high = _bootstrap_ci(finite_paired, bootstrap_samples, seed + 1000 + metric_pos)
        statistic = float("nan")
        p_value = float("nan")
        if wilcoxon is not None and finite_paired.size:
            try:
                result = wilcoxon(finite_paired, zero_method="zsplit")
                statistic = float(result.statistic)
                p_value = float(result.pvalue)
            except ValueError:
                pass
        rows.append(
            {
                "metric": metric,
                "comparison": "treatment_minus_matched_control",
                "n": int(finite_paired.size),
                "effect_mean": _safe_mean(finite_paired),
                "effect_median": _safe_median(finite_paired),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "test": "paired_wilcoxon_signed_rank",
                "statistic": statistic,
                "p_value": p_value,
            }
        )
    return pd.DataFrame(rows)


def fit_regressions(table: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    try:
        import statsmodels.formula.api as smf
    except Exception as exc:
        return pd.DataFrame(
            [
                {
                    "outcome": "delta_rashomon_capacity",
                    "term": "statsmodels_unavailable",
                    "coefficient": float("nan"),
                    "std_error": float("nan"),
                    "p_value": float("nan"),
                    "note": str(exc),
                }
            ]
        )

    formulas = {
        "delta_rashomon_capacity": (
            "delta_rashomon_capacity ~ treatment_indicator + "
            "delta_neighborhood_label_entropy + original_degree + original_confidence + "
            "original_correct + C(pair_id)"
        ),
        "delta_predictive_entropy": (
            "delta_predictive_entropy ~ treatment_indicator + "
            "delta_neighborhood_label_entropy + original_degree + original_confidence + "
            "original_correct + C(pair_id)"
        ),
    }
    cluster_count = int(table["pair_id"].nunique())
    for outcome, formula in formulas.items():
        model = smf.ols(formula, data=table).fit(
            cov_type="cluster",
            cov_kwds={
                "groups": table["pair_id"],
                "use_correction": True,
                "df_correction": True,
            },
            use_t=True,
        )
        reported_terms = [term for term in model.params.index if not term.startswith("C(pair_id)")]
        for term in reported_terms:
            rows.append(
                {
                    "outcome": outcome,
                    "term": term,
                    "coefficient": float(model.params[term]),
                    "std_error": float(model.bse[term]),
                    "p_value": float(model.pvalues[term]),
                    "r_squared": float(model.rsquared),
                    "n": int(model.nobs),
                    "cluster_count": cluster_count,
                    "reference_df": cluster_count - 1,
                    "note": (
                        "matched-pair fixed effects; pair-clustered standard errors; "
                        "small-sample t reference with df=clusters-1"
                    ),
                }
            )
    return pd.DataFrame(rows)


def write_plots(table: pd.DataFrame, output_prefix: Path) -> list[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping treatment/control plots because matplotlib is unavailable: {exc}")
        return []

    paths: list[Path] = []
    groups = ["control", "treatment"]
    colors = {"control": "#4C78A8", "treatment": "#F58518"}

    box_path = output_prefix.with_suffix(".delta_boxplots.png")
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for ax, metric, ylabel in [
        (axes[0], "delta_neighborhood_label_entropy", "Delta neighborhood label entropy"),
        (axes[1], "delta_rashomon_capacity", "Delta Rashomon capacity"),
    ]:
        values = [
            table.loc[table["group"] == group, metric].to_numpy(dtype=float) for group in groups
        ]
        bp = ax.boxplot(values, tick_labels=groups, patch_artist=True, showfliers=False)
        for patch, group in zip(bp["boxes"], groups, strict=True):
            patch.set_facecolor(colors[group])
            patch.set_alpha(0.55)
        ax.axhline(0.0, color="black", linewidth=1)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(box_path, dpi=180)
    plt.close(fig)
    paths.append(box_path)

    scatter_path = output_prefix.with_suffix(".delta_entropy_capacity.png")
    fig, ax = plt.subplots(figsize=(6, 4))
    for group in groups:
        subset = table[table["group"] == group]
        ax.scatter(
            subset["delta_neighborhood_label_entropy"],
            subset["delta_rashomon_capacity"],
            s=28,
            alpha=0.75,
            label=group,
            color=colors[group],
        )
    ax.axhline(0.0, color="black", linewidth=1)
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Delta neighborhood label entropy")
    ax.set_ylabel("Delta Rashomon capacity")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(scatter_path, dpi=180)
    plt.close(fig)
    paths.append(scatter_path)
    return paths


def main() -> None:
    args = parse_args()
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    metadata = _read_metadata(args.rewiring_metadata)
    original_structure = pd.read_csv(args.original_structure)
    rewired_structure = pd.read_csv(args.rewired_structure)
    delta_table = pd.read_csv(args.delta_table)

    table = build_treatment_control_table(
        metadata, original_structure, rewired_structure, delta_table
    )
    summary = summarize_groups(table, bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    tests = run_tests(table, bootstrap_samples=args.bootstrap_samples, seed=args.seed)
    regressions = fit_regressions(table)

    nodes_path = output_prefix.with_suffix(".nodes.csv")
    summary_path = output_prefix.with_suffix(".summary.csv")
    tests_path = output_prefix.with_suffix(".tests.csv")
    regression_path = output_prefix.with_suffix(".regression.csv")
    table.to_csv(nodes_path, index=False)
    summary.to_csv(summary_path, index=False)
    tests.to_csv(tests_path, index=False)
    regressions.to_csv(regression_path, index=False)
    plot_paths = write_plots(table, output_prefix)

    treatment = table[table["group"] == "treatment"]
    control = table[table["group"] == "control"]
    print(f"Wrote treatment/control node table to {nodes_path}")
    print(f"Wrote treatment/control summary to {summary_path}")
    print(f"Wrote treatment/control tests to {tests_path}")
    print(f"Wrote treatment/control regressions to {regression_path}")
    for path in plot_paths:
        print(f"Wrote treatment/control plot to {path}")
    print(
        "delta_capacity_mean "
        f"treatment={treatment['delta_rashomon_capacity'].mean():.6f} "
        f"control={control['delta_rashomon_capacity'].mean():.6f}"
    )
    print(
        "delta_entropy_mean "
        f"treatment={treatment['delta_neighborhood_label_entropy'].mean():.6f} "
        f"control={control['delta_neighborhood_label_entropy'].mean():.6f}"
    )


if __name__ == "__main__":
    main()
