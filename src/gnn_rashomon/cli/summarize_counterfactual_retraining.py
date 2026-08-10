from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from gnn_rashomon.analysis.counterfactual_retraining import (
    BASELINE_SET_TEMPLATE,
    CONDITIONS,
    DATASETS,
    EPSILON,
    GRAPH_SEEDS,
    MAX_EPOCHS,
    OUTPUT_ROOT,
    graph_path_for,
    metadata_path_for,
    rashomon_type_for,
    set_type_for,
)
from gnn_rashomon.analysis.multiplicity_from_set import multiplicity_metrics_from_set
from gnn_rashomon.rashomon.io import read_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize counterfactual retraining Rashomon sets versus the original-graph baseline."
        )
    )
    parser.add_argument(
        "--output-root",
        default=str(OUTPUT_ROOT),
        help="Root used for cf-retraining metrics/rashomon_sets.",
    )
    parser.add_argument(
        "--baseline-root",
        default="outputs",
        help="Root containing original-graph Rashomon sets and metrics.",
    )
    parser.add_argument(
        "--output-prefix",
        default="outputs/counterfactual_retraining/cf_retrain_summary",
    )
    return parser.parse_args()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _metrics_row(
    *,
    dataset: str,
    condition_id: str,
    graph_seed: int | None,
    set_path: Path,
    graph_path: Path | None,
    project_root: Path,
) -> dict[str, object]:
    payload = read_json(set_path)
    metrics = multiplicity_metrics_from_set(payload, project_root=project_root)
    return {
        "dataset": dataset,
        "condition_id": condition_id,
        "graph_seed": graph_seed,
        "set_id": payload["set_id"],
        "set_path": str(set_path),
        "graph_path": str(graph_path) if graph_path is not None else None,
        "candidate_count": int(payload.get("candidate_count", 0)),
        **metrics,
    }


def main() -> None:
    args = parse_args()
    project_root = _project_root()
    output_root = Path(args.output_root)
    baseline_root = Path(args.baseline_root)
    rows: list[dict[str, object]] = []

    for dataset in DATASETS:
        baseline_set = (
            baseline_root
            / "rashomon_sets"
            / f"{BASELINE_SET_TEMPLATE.format(dataset=dataset)}.json"
        )
        if not baseline_set.exists():
            raise SystemExit(f"Missing baseline Rashomon set: {baseline_set}")
        rows.append(
            _metrics_row(
                dataset=dataset,
                condition_id="original",
                graph_seed=None,
                set_path=baseline_set,
                graph_path=None,
                project_root=project_root,
            )
        )
        for condition in CONDITIONS:
            for graph_seed in GRAPH_SEEDS:
                set_type = set_type_for(condition, graph_seed)
                set_path = (
                    output_root
                    / "rashomon_sets"
                    / f"{dataset}-gcn-{set_type}-relative{EPSILON:g}-epochs{MAX_EPOCHS}.json"
                )
                if not set_path.exists():
                    continue
                rows.append(
                    _metrics_row(
                        dataset=dataset,
                        condition_id=condition.condition_id,
                        graph_seed=graph_seed,
                        set_path=set_path,
                        graph_path=graph_path_for(dataset, condition, graph_seed),
                        project_root=project_root,
                    )
                )

    detail = pd.DataFrame(rows)
    if detail.empty:
        raise SystemExit("No counterfactual retraining Rashomon sets found.")

    baseline = detail[detail["condition_id"] == "original"][
        [
            "dataset",
            "retained_count",
            "fraction_prediction_disagreement",
            "mean_probability_diameter",
            "mean_pairwise_total_variation",
            "validation_accuracy_mean",
        ]
    ].rename(
        columns={
            "retained_count": "baseline_retained_count",
            "fraction_prediction_disagreement": "baseline_disagreement",
            "mean_probability_diameter": "baseline_mean_diameter",
            "mean_pairwise_total_variation": "baseline_mean_pairwise_tv",
            "validation_accuracy_mean": "baseline_val_acc_mean",
        }
    )
    treated = detail[detail["condition_id"] != "original"].merge(baseline, on="dataset", how="left")
    for metric, baseline_col in [
        ("retained_count", "baseline_retained_count"),
        ("fraction_prediction_disagreement", "baseline_disagreement"),
        ("mean_probability_diameter", "baseline_mean_diameter"),
        ("mean_pairwise_total_variation", "baseline_mean_pairwise_tv"),
        ("validation_accuracy_mean", "baseline_val_acc_mean"),
    ]:
        treated[f"delta_{metric}"] = treated[metric] - treated[baseline_col]

    condition_summary = (
        treated.groupby(["dataset", "condition_id"], dropna=False, sort=True)
        .agg(
            n_graphs=("graph_seed", "count"),
            mean_retained=("retained_count", "mean"),
            mean_disagreement=("fraction_prediction_disagreement", "mean"),
            mean_diameter=("mean_probability_diameter", "mean"),
            mean_pairwise_tv=("mean_pairwise_total_variation", "mean"),
            mean_val_acc=("validation_accuracy_mean", "mean"),
            mean_delta_retained=("delta_retained_count", "mean"),
            mean_delta_disagreement=("delta_fraction_prediction_disagreement", "mean"),
            mean_delta_diameter=("delta_mean_probability_diameter", "mean"),
            mean_delta_pairwise_tv=("delta_mean_pairwise_total_variation", "mean"),
            fraction_graphs_higher_disagreement=(
                "delta_fraction_prediction_disagreement",
                lambda s: float((s > 0).mean()),
            ),
            fraction_graphs_higher_diameter=(
                "delta_mean_probability_diameter",
                lambda s: float((s > 0).mean()),
            ),
        )
        .reset_index()
    )

    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    detail_path = prefix.with_name(f"{prefix.name}_detail.csv")
    treated_path = prefix.with_name(f"{prefix.name}_vs_baseline.csv")
    summary_path = prefix.with_name(f"{prefix.name}_by_condition.csv")
    meta_path = prefix.with_name(f"{prefix.name}_meta.json")
    detail.to_csv(detail_path, index=False)
    treated.to_csv(treated_path, index=False)
    condition_summary.to_csv(summary_path, index=False)
    meta = {
        "question": "Does R(G') exhibit systematically different multiplicity from R(G)?",
        "datasets": list(DATASETS),
        "conditions": [condition.condition_id for condition in CONDITIONS],
        "graph_seeds": list(GRAPH_SEEDS),
        "epsilon": EPSILON,
        "max_epochs": MAX_EPOCHS,
        "output_root": str(output_root),
        "baseline_root": str(baseline_root),
        "detail_csv": str(detail_path),
        "vs_baseline_csv": str(treated_path),
        "by_condition_csv": str(summary_path),
        "graph_sources": {
            condition.condition_id: [
                str(graph_path_for(dataset, condition, seed))
                for dataset in DATASETS
                for seed in GRAPH_SEEDS
            ]
            for condition in CONDITIONS
        },
        "metadata_sources": {
            condition.condition_id: [
                str(metadata_path_for(dataset, condition, seed))
                for dataset in DATASETS
                for seed in GRAPH_SEEDS
            ]
            for condition in CONDITIONS
        },
        "rashomon_type_examples": [
            rashomon_type_for(CONDITIONS[0], 0),
            rashomon_type_for(CONDITIONS[1], 0),
            rashomon_type_for(CONDITIONS[2], 0),
        ],
    }
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"detail_rows": len(detail), "condition_rows": len(condition_summary)}, indent=2))


if __name__ == "__main__":
    main()
