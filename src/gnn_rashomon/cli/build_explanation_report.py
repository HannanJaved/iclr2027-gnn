from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests


METRICS = [
    "top_k_jaccard_mean",
    "spearman_mean",
    "kendall_mean",
    "explanation_size_mean",
]

PREDICTORS = [
    "rashomon_capacity",
    "predictive_entropy",
    "variation_ratio",
    "degree",
    "confidence",
    "correct",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper-ready explanation-instability report tables.")
    parser.add_argument("--metadata", required=True, help="Explanation analysis metadata JSON.")
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Defaults to outputs/figures/<metadata-stem>.",
    )
    return parser.parse_args()


def _bootstrap_ci(values: np.ndarray, samples: int = 2000, seed: int = 0) -> tuple[float, float]:
    clean = values[np.isfinite(values)]
    if clean.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.choice(clean, size=(samples, clean.size), replace=True)
    means = draws.mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _out(prefix: Path, suffix: str) -> Path:
    return Path(f"{prefix}{suffix}")


def _group_summary(nodes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric_pos, metric in enumerate(METRICS):
        for group, subset in nodes.groupby("multiplicity_group", sort=True):
            values = subset[metric].to_numpy(dtype=float)
            low, high = _bootstrap_ci(values, seed=metric_pos)
            rows.append(
                {
                    "metric": metric,
                    "multiplicity_group": group,
                    "n": int(np.isfinite(values).sum()),
                    "mean": float(np.nanmean(values)),
                    "median": float(np.nanmedian(values)),
                    "std": float(np.nanstd(values, ddof=1)),
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                }
            )
    return pd.DataFrame(rows)


def _correlation_table(nodes: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for metric in METRICS:
        for predictor in PREDICTORS:
            pair = nodes[[metric, predictor]].dropna()
            if len(pair) < 3:
                statistic = float("nan")
                p_value = float("nan")
            else:
                result = spearmanr(pair[predictor], pair[metric])
                statistic = float(result.statistic)
                p_value = float(result.pvalue)
            rows.append(
                {
                    "stability_metric": metric,
                    "predictor": predictor,
                    "spearman": statistic,
                    "p_value": p_value,
                    "n": int(len(pair)),
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


def _paper_summary(metadata: dict[str, object], nodes: pd.DataFrame, pairwise: pd.DataFrame) -> pd.DataFrame:
    high = nodes[nodes["multiplicity_group"] == "high"]
    low = nodes[nodes["multiplicity_group"] == "low"]
    corr = spearmanr(nodes["rashomon_capacity"], nodes["top_k_jaccard_mean"])
    return pd.DataFrame(
        [
            {
                "dataset": metadata.get("dataset"),
                "explainer": metadata.get("explainer"),
                "selected_nodes": int(len(nodes)),
                "selected_models": int(metadata.get("selected_model_count", 0)),
                "model_pairs_per_node": int(pairwise.groupby("node_id").size().median()),
                "top_k": int(metadata.get("top_k", 0)),
                "explainer_epochs": int(metadata.get("explainer_epochs", 0)),
                "mean_top_k_jaccard": float(nodes["top_k_jaccard_mean"].mean()),
                "median_top_k_jaccard": float(nodes["top_k_jaccard_mean"].median()),
                "high_capacity_jaccard_mean": float(high["top_k_jaccard_mean"].mean()),
                "low_capacity_jaccard_mean": float(low["top_k_jaccard_mean"].mean()),
                "high_minus_low_jaccard_mean": float(
                    high["top_k_jaccard_mean"].mean() - low["top_k_jaccard_mean"].mean()
                ),
                "capacity_jaccard_spearman": float(corr.statistic),
                "capacity_jaccard_p_value": float(corr.pvalue),
            }
        ]
    )


def _write_plots(nodes: pd.DataFrame, correlations: pd.DataFrame, prefix: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping explanation report plots because matplotlib is unavailable: {exc}", flush=True)
        return

    groups = ["low", "medium", "high"]
    colors = ["#4C78A8", "#54A24B", "#F58518"]
    fig, ax = plt.subplots(figsize=(6, 4))
    data = [
        nodes.loc[nodes["multiplicity_group"] == group, "top_k_jaccard_mean"].dropna().to_numpy(dtype=float)
        for group in groups
    ]
    bp = ax.boxplot(data, tick_labels=groups, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_xlabel("Multiplicity group")
    ax.set_ylabel("Mean top-k explanation Jaccard")
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(_out(prefix, ".jaccard_by_multiplicity_group.png"), dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(
        nodes["rashomon_capacity"].to_numpy(dtype=float),
        nodes["top_k_jaccard_mean"].to_numpy(dtype=float),
        s=32,
        alpha=0.75,
    )
    ax.set_xlabel("Rashomon capacity")
    ax.set_ylabel("Mean top-k explanation Jaccard")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(_out(prefix, ".capacity_vs_jaccard.png"), dpi=180)
    plt.close(fig)

    heat = correlations.pivot(index="predictor", columns="stability_metric", values="spearman")
    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(heat.to_numpy(dtype=float), vmin=-1, vmax=1, cmap="coolwarm")
    ax.set_xticks(np.arange(len(heat.columns)), labels=heat.columns, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(heat.index)), labels=heat.index)
    ax.set_title("Spearman correlations")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(_out(prefix, ".correlation_heatmap.png"), dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    outputs = metadata["outputs"]
    nodes = pd.read_csv(outputs["node_summary"])
    pairwise = pd.read_csv(outputs["pairwise"])

    if args.output_prefix:
        prefix = Path(args.output_prefix)
    else:
        prefix = Path("outputs/figures") / metadata_path.name.removesuffix(".metadata.json")
    prefix.parent.mkdir(parents=True, exist_ok=True)

    group_summary = _group_summary(nodes)
    correlations = _correlation_table(nodes)
    paper_summary = _paper_summary(metadata, nodes, pairwise)

    group_path = _out(prefix, ".group_summary.csv")
    corr_path = _out(prefix, ".correlations.csv")
    paper_path = _out(prefix, ".paper_summary.csv")
    group_summary.to_csv(group_path, index=False)
    correlations.to_csv(corr_path, index=False)
    paper_summary.to_csv(paper_path, index=False)
    _write_plots(nodes, correlations, prefix)

    print(f"Wrote explanation group summary to {group_path}")
    print(f"Wrote explanation correlations to {corr_path}")
    print(f"Wrote explanation paper summary to {paper_path}")
    row = paper_summary.iloc[0]
    print(
        "capacity_vs_jaccard "
        f"spearman={row['capacity_jaccard_spearman']:.4f} "
        f"p={row['capacity_jaccard_p_value']:.4g} "
        f"high_minus_low={row['high_minus_low_jaccard_mean']:.4f}"
    )


if __name__ == "__main__":
    main()
