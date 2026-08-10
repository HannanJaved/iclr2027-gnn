from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


METRICS = [
    "delta_predictive_entropy",
    "delta_variation_ratio",
    "delta_rashomon_capacity",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build counterfactual rewiring report tables and plots.")
    parser.add_argument("--dataset", default="cora")
    parser.add_argument("--mode", default="random")
    parser.add_argument("--strength-ids", nargs="+", default=["0p01", "0p05", "0p1", "0p25"])
    parser.add_argument("--strength-values", nargs="+", type=float, default=[0.01, 0.05, 0.10, 0.25])
    parser.add_argument(
        "--graph-ids",
        nargs="+",
        help="Optional graph file IDs after '<dataset>-'. Defaults to '<mode>-<strength_id>'.",
    )
    parser.add_argument(
        "--metric-ids",
        nargs="+",
        help="Optional metric IDs after '<dataset>-gcn-'. Defaults to 'rewired_<mode>_<strength_id>'.",
    )
    parser.add_argument(
        "--delta-ids",
        nargs="+",
        help="Optional delta file IDs after '<dataset>-'. Defaults to '<mode>-<strength_id>-seed0'.",
    )
    parser.add_argument("--plot-prefix", help="Output plot/table prefix. Defaults to '<dataset>_<mode>_rewiring'.")
    parser.add_argument("--x-label", help="X-axis label for generated plots.")
    parser.add_argument("--plot-only", action="store_true", help="Read existing report tables and regenerate plots only.")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument(
        "--set-suffix",
        default="relative0.1-epochs200",
        help="Metric file suffix after '<dataset>-gcn-<metric_id>-'.",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bootstrap_ci(values: np.ndarray, samples: int, seed: int) -> tuple[float, float]:
    import numpy as np

    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype=float)
    for idx in range(samples):
        draw = rng.choice(values, size=values.size, replace=True)
        means[idx] = float(np.mean(draw))
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _maybe_plot(dose: pd.DataFrame, stats: pd.DataFrame, figures_dir: Path, plot_prefix: str, x_label: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - depends on optional plotting install
        print(f"Skipping plots because matplotlib is unavailable: {exc}")
        return

    plots = [
        ("rewired_homophily", "Rewired Homophily", f"{plot_prefix}_homophily.png"),
        ("mean_neighborhood_label_entropy", "Mean Neighborhood Label Entropy", f"{plot_prefix}_neighborhood_entropy.png"),
        ("mean_confidence", "Mean Confidence", f"{plot_prefix}_confidence.png"),
        ("mean_correct", "Mean Correctness", f"{plot_prefix}_correctness.png"),
        ("disagreement_fraction", "Prediction Disagreement Fraction", f"{plot_prefix}_disagreement.png"),
        ("capacity_mean", "Mean Probability Diameter", f"{plot_prefix}_capacity.png"),
    ]
    for column, ylabel, filename in plots:
        print(f"Writing plot {filename}", flush=True)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(dose["strength"], dose[column], marker="o")
        ax.set_xlabel(x_label)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(figures_dir / filename, dpi=180)
        plt.close(fig)

    print(f"Writing plot {plot_prefix}_delta_metrics.png", flush=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    for metric, label in [
        ("delta_predictive_entropy", "Predictive Entropy"),
        ("delta_variation_ratio", "Variation Ratio"),
        ("delta_rashomon_capacity", "Rashomon Capacity"),
    ]:
        subset = stats[stats["metric"] == metric]
        ax.errorbar(
            subset["strength"],
            subset["mean"],
            yerr=[subset["mean"] - subset["bootstrap_ci_low"], subset["bootstrap_ci_high"] - subset["mean"]],
            marker="o",
            capsize=3,
            label=label,
        )
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Mean node-level delta")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures_dir / f"{plot_prefix}_delta_metrics.png", dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if len(args.strength_ids) != len(args.strength_values):
        raise SystemExit("--strength-ids and --strength-values must have the same length.")
    graph_ids = args.graph_ids or [f"{args.mode}-{strength_id}" for strength_id in args.strength_ids]
    metric_ids = args.metric_ids or [f"rewired_{args.mode}_{strength_id}" for strength_id in args.strength_ids]
    delta_ids = args.delta_ids or [f"{args.mode}-{strength_id}-seed0" for strength_id in args.strength_ids]
    for label, values in [("--graph-ids", graph_ids), ("--metric-ids", metric_ids), ("--delta-ids", delta_ids)]:
        if len(values) != len(args.strength_ids):
            raise SystemExit(f"{label} must have the same length as --strength-ids.")
    plot_prefix = args.plot_prefix or f"{args.dataset}_{args.mode}_rewiring"
    x_label = args.x_label or f"{args.mode.replace('_', ' ').title()} rewiring strength"

    output_dir = Path(args.output_dir)
    metrics_dir = output_dir / "metrics"
    graphs_dir = output_dir / "graphs"
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    dose_path = figures_dir / f"{plot_prefix}_dose_response.csv"
    stats_path = figures_dir / f"{plot_prefix}_delta_tests.csv"

    if args.plot_only:
        print(f"Reading existing report tables {dose_path} and {stats_path}", flush=True)
        import pandas as pd

        dose = pd.read_csv(dose_path)
        stats = pd.read_csv(stats_path)
        print("Writing report plots", flush=True)
        _maybe_plot(dose, stats, figures_dir, plot_prefix, x_label)
        print(f"Wrote plots with prefix {plot_prefix}", flush=True)
        return

    dose_rows: list[dict[str, float | int | str]] = []
    stat_rows: list[dict[str, float | int | str]] = []
    print("Importing report dependencies", flush=True)
    import numpy as np
    import pandas as pd

    for pos, (strength_id, strength, graph_id, metric_id, delta_id) in enumerate(
        zip(args.strength_ids, args.strength_values, graph_ids, metric_ids, delta_ids)
    ):
        print(f"Reading inputs for {strength_id}", flush=True)
        graph = _read_json(graphs_dir / f"{args.dataset}-{graph_id}.json")
        metric_prefix = f"{args.dataset}-gcn-{metric_id}-{args.set_suffix}"
        mult = _read_json(metrics_dir / f"{metric_prefix}.multiplicity_summary.json")
        structure = _read_json(
            metrics_dir / f"{metric_prefix}.structure_summary.json"
        )
        delta = pd.read_csv(metrics_dir / f"{args.dataset}-{delta_id}.delta_multiplicity.csv")

        dose_rows.append(
            {
                "dataset": args.dataset,
                "mode": args.mode,
                "strength_id": strength_id,
                "strength": strength,
                "graph_id": graph_id,
                "metric_id": metric_id,
                "delta_id": delta_id,
                "accepted_swaps": int(graph["accepted_swaps"]),
                "attempted_swaps": int(graph["attempted_swaps"]),
                "original_homophily": float(graph["original_homophily"]),
                "rewired_homophily": float(graph["final_homophily"]),
                "retained_count": int(mult["retained_count"]),
                "disagreement_fraction": float(mult["fraction_prediction_disagreement"]),
                "entropy_mean": float(mult["predictive_entropy"]["mean"]),
                "variation_ratio_mean": float(mult["variation_ratio"]["mean"]),
                "capacity_mean": float(mult["rashomon_capacity"]["mean"]),
                "mean_confidence": float(structure["mean_confidence"]),
                "mean_correct": float(structure["mean_correct"]),
                "mean_neighborhood_label_entropy": float(structure["mean_neighborhood_label_entropy"]),
                "mean_local_homophily": float(structure["mean_local_homophily"]),
            }
        )

        for metric in METRICS:
            print(f"Computing delta statistics for {strength_id} {metric}", flush=True)
            values = delta[metric].to_numpy(dtype=float)
            low, high = _bootstrap_ci(values, args.bootstrap_samples, args.seed + pos)
            try:
                from scipy.stats import wilcoxon

                test = wilcoxon(values, zero_method="zsplit")
                statistic = float(test.statistic)
                p_value = float(test.pvalue)
            except ValueError:
                statistic = float("nan")
                p_value = float("nan")
            stat_rows.append(
                {
                    "dataset": args.dataset,
                    "mode": args.mode,
                    "strength_id": strength_id,
                    "strength": strength,
                    "metric": metric,
                    "n": int(values.size),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std": float(np.std(values, ddof=1)),
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "wilcoxon_statistic": statistic,
                    "wilcoxon_p_value": p_value,
                }
            )

    dose = pd.DataFrame(dose_rows)
    stats = pd.DataFrame(stat_rows)
    print("Writing report tables", flush=True)
    dose.to_csv(dose_path, index=False)
    stats.to_csv(stats_path, index=False)
    print("Writing report plots", flush=True)
    _maybe_plot(dose, stats, figures_dir, plot_prefix, x_label)
    print(f"Wrote dose-response table to {dose_path}")
    print(f"Wrote paired delta tests to {stats_path}")


if __name__ == "__main__":
    main()
