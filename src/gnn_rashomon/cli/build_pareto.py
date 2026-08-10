from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from gnn_rashomon.analysis.pareto import (
    model_explanation_stability,
    pareto_frontier,
    selection_baselines,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Pareto frontiers from saved Rashomon artifacts."
    )
    parser.add_argument("--mode", choices=["explanation", "fairness"], default="explanation")
    parser.add_argument("--metadata", help="Explanation analysis metadata JSON.")
    parser.add_argument("--fairness-table", help="Fairness model table CSV for fairness mode.")
    parser.add_argument(
        "--fairness-metric",
        choices=["demographic_parity_gap", "equalized_odds_gap", "degree_disparity"],
        default="demographic_parity_gap",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Defaults to outputs/figures/<metadata-stem>.pareto",
    )
    return parser.parse_args()


def _out(prefix: Path, suffix: str) -> Path:
    return Path(f"{prefix}{suffix}")


def _write_explanation_plot(models: pd.DataFrame, prefix: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping Pareto plot because matplotlib is unavailable: {exc}", flush=True)
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    dominated = models[~models["is_pareto_efficient"]]
    frontier = models[models["is_pareto_efficient"]]
    ax.scatter(
        dominated["validation_accuracy"],
        dominated["mean_top_k_jaccard"],
        s=36,
        alpha=0.55,
        label="Dominated",
        color="#4C78A8",
    )
    ax.scatter(
        frontier["validation_accuracy"],
        frontier["mean_top_k_jaccard"],
        s=52,
        alpha=0.9,
        label="Pareto efficient",
        color="#F58518",
    )
    if len(frontier) > 1:
        ordered = frontier.sort_values("validation_accuracy")
        ax.plot(
            ordered["validation_accuracy"],
            ordered["mean_top_k_jaccard"],
            color="#F58518",
            linewidth=1.5,
            alpha=0.8,
        )
    ax.set_xlabel("Validation accuracy")
    ax.set_ylabel("Mean top-k explanation Jaccard")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(_out(prefix, ".validation_vs_explanation_stability.png"), dpi=180)
    plt.close(fig)


def _write_fairness_plot(models: pd.DataFrame, prefix: Path, fairness_metric: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"Skipping Pareto plot because matplotlib is unavailable: {exc}", flush=True)
        return

    selection_metric = f"validation_{fairness_metric}"
    fig, ax = plt.subplots(figsize=(6, 4))
    dominated = models[~models["is_pareto_efficient"]]
    frontier = models[models["is_pareto_efficient"]]
    ax.scatter(
        dominated["validation_accuracy"],
        dominated[selection_metric],
        s=36,
        alpha=0.55,
        label="Dominated",
        color="#4C78A8",
    )
    ax.scatter(
        frontier["validation_accuracy"],
        frontier[selection_metric],
        s=52,
        alpha=0.9,
        label="Pareto efficient",
        color="#F58518",
    )
    if len(frontier) > 1:
        ordered = frontier.sort_values("validation_accuracy")
        ax.plot(
            ordered["validation_accuracy"],
            ordered[selection_metric],
            color="#F58518",
            linewidth=1.5,
            alpha=0.8,
        )
    ax.set_xlabel("Validation accuracy")
    ax.set_ylabel(fairness_metric.replace("_", " ").title())
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(_out(prefix, f".validation_vs_{fairness_metric}.png"), dpi=180)
    plt.close(fig)


def build_explanation_pareto(metadata_path: Path, output_prefix: Path) -> None:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    outputs = metadata["outputs"]
    pairwise = pd.read_csv(outputs["pairwise"])
    selected_models = pd.read_csv(outputs["selected_models"])
    stability = model_explanation_stability(pairwise)
    models = selected_models.merge(stability, on="run_id", how="left", validate="one_to_one")
    models = pareto_frontier(
        models,
        objective_columns=["validation_accuracy", "mean_top_k_jaccard"],
        maximize=[True, True],
    )
    models = models.sort_values(
        ["is_pareto_efficient", "validation_accuracy", "mean_top_k_jaccard"],
        ascending=[False, False, False],
    )
    baselines = selection_baselines(models)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    model_path = _out(output_prefix, ".models.csv")
    baseline_path = _out(output_prefix, ".baselines.csv")
    summary_path = _out(output_prefix, ".summary.json")
    models.to_csv(model_path, index=False)
    baselines.to_csv(baseline_path, index=False)
    summary = {
        "mode": "explanation",
        "metadata": str(metadata_path),
        "selected_model_count": int(len(models)),
        "pareto_model_count": int(models["is_pareto_efficient"].sum()),
        "objective_x": "validation_accuracy",
        "objective_y": "mean_top_k_jaccard",
        "selection_uses_test_metrics": False,
        "test_accuracy_is_report_only": True,
        "model_table": str(model_path),
        "baseline_table": str(baseline_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_explanation_plot(models, output_prefix)
    print(f"Wrote Pareto model table to {model_path}")
    print(f"Wrote Pareto baseline table to {baseline_path}")
    print(f"Wrote Pareto summary to {summary_path}")
    print(
        "pareto "
        f"models={len(models)} frontier={int(models['is_pareto_efficient'].sum())} "
        f"best_val={models['validation_accuracy'].max():.4f} "
        f"best_stability={models['mean_top_k_jaccard'].max():.4f}"
    )


def _fairness_selection_baselines(models: pd.DataFrame, fairness_metric: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if models.empty:
        return pd.DataFrame()

    def add(name: str, row: pd.Series) -> None:
        selection_metric = f"validation_{fairness_metric}"
        report_metric = f"test_{fairness_metric}_report_only"
        rows.append(
            {
                "selection_rule": name,
                "run_id": row["run_id"],
                "validation_accuracy": row.get("validation_accuracy"),
                "test_accuracy_report_only": row.get("test_accuracy_report_only"),
                selection_metric: row.get(selection_metric),
                report_metric: row.get(report_metric),
                "is_pareto_efficient": row.get("is_pareto_efficient"),
            }
        )

    add(
        "best_validation_accuracy",
        models.sort_values("validation_accuracy", ascending=False).iloc[0],
    )
    selection_metric = f"validation_{fairness_metric}"
    add(f"best_{fairness_metric}", models.sort_values(selection_metric, ascending=True).iloc[0])
    add(
        "first_pareto_model",
        models.sort_values(
            ["is_pareto_efficient", "validation_accuracy", selection_metric],
            ascending=[False, False, True],
        ).iloc[0],
    )
    return pd.DataFrame(rows)


def build_fairness_pareto(
    fairness_table_path: Path, output_prefix: Path, fairness_metric: str
) -> None:
    table = pd.read_csv(fairness_table_path)
    selection_metric = f"validation_{fairness_metric}"
    report_metric = f"test_{fairness_metric}_report_only"
    required = {"validation_accuracy", selection_metric, report_metric}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(
            "Fairness table lacks split-specific columns: " + ", ".join(sorted(missing))
        )
    models = pareto_frontier(
        table,
        objective_columns=["validation_accuracy", selection_metric],
        maximize=[True, False],
    )
    models = models.sort_values(
        ["is_pareto_efficient", "validation_accuracy", selection_metric],
        ascending=[False, False, True],
    )
    baselines = _fairness_selection_baselines(models, fairness_metric)

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    model_path = _out(output_prefix, ".models.csv")
    baseline_path = _out(output_prefix, ".baselines.csv")
    summary_path = _out(output_prefix, ".summary.json")
    models.to_csv(model_path, index=False)
    baselines.to_csv(baseline_path, index=False)
    summary = {
        "mode": "fairness",
        "fairness_table": str(fairness_table_path),
        "fairness_metric": fairness_metric,
        "selected_model_count": int(len(models)),
        "pareto_model_count": int(models["is_pareto_efficient"].sum()),
        "objective_x": "validation_accuracy",
        "objective_y": selection_metric,
        "report_only_metric": report_metric,
        "selection_uses_test_metrics": False,
        "test_accuracy_is_report_only": True,
        "model_table": str(model_path),
        "baseline_table": str(baseline_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    _write_fairness_plot(models, output_prefix, fairness_metric)
    print(f"Wrote Pareto model table to {model_path}")
    print(f"Wrote Pareto baseline table to {baseline_path}")
    print(f"Wrote Pareto summary to {summary_path}")
    print(
        "pareto "
        f"models={len(models)} frontier={int(models['is_pareto_efficient'].sum())} "
        f"best_val={models['validation_accuracy'].max():.4f} "
        f"best_{fairness_metric}={models[selection_metric].min():.4f}"
    )


def main() -> None:
    args = parse_args()
    metadata_path = Path(args.metadata) if args.metadata else None
    if args.output_prefix:
        output_prefix = Path(args.output_prefix)
    elif args.mode == "fairness":
        if not args.fairness_table:
            raise SystemExit("--fairness-table is required for fairness mode.")
        table_path = Path(args.fairness_table)
        table_stem = table_path.name.removesuffix(".fairness_models.csv")
        output_prefix = Path("outputs/figures") / (
            f"{table_stem}.fairness_{args.fairness_metric}.pareto"
        )
    else:
        if metadata_path is None:
            raise SystemExit("--metadata is required for explanation mode.")
        output_prefix = (
            Path("outputs/figures") / f"{metadata_path.name.removesuffix('.metadata.json')}.pareto"
        )
    if args.mode == "explanation":
        if metadata_path is None:
            raise SystemExit("--metadata is required for explanation mode.")
        build_explanation_pareto(metadata_path, output_prefix)
    elif args.mode == "fairness":
        if not args.fairness_table:
            raise SystemExit("--fairness-table is required for fairness mode.")
        build_fairness_pareto(Path(args.fairness_table), output_prefix, args.fairness_metric)


if __name__ == "__main__":
    main()
