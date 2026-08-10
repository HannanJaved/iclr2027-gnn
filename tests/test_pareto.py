import json

import pandas as pd
import pytest

from gnn_rashomon.analysis.pareto import (
    model_explanation_stability,
    pareto_frontier,
    selection_baselines,
)
from gnn_rashomon.cli.build_pareto import build_fairness_pareto


def test_pareto_frontier_marks_dominated_rows():
    frame = pd.DataFrame(
        {
            "run_id": ["a", "b", "c", "d"],
            "validation_accuracy": [0.80, 0.82, 0.79, 0.81],
            "mean_top_k_jaccard": [0.60, 0.55, 0.50, 0.70],
        }
    )

    out = pareto_frontier(
        frame,
        objective_columns=["validation_accuracy", "mean_top_k_jaccard"],
        maximize=[True, True],
    )

    efficient = set(out.loc[out["is_pareto_efficient"], "run_id"])
    assert efficient == {"b", "d"}
    assert out.loc[out["run_id"] == "c", "dominated_by"].iloc[0] in {"a", "b", "d"}


def test_model_explanation_stability_aggregates_both_pairwise_sides():
    pairwise = pd.DataFrame(
        {
            "left_run_id": ["a", "a", "b"],
            "right_run_id": ["b", "c", "c"],
            "top_k_jaccard": [0.8, 0.4, 0.6],
            "spearman": [0.9, 0.5, 0.7],
            "kendall": [0.8, 0.4, 0.6],
            "left_explanation_size": [3, 3, 4],
            "right_explanation_size": [4, 5, 5],
        }
    )

    stability = model_explanation_stability(pairwise)

    row_a = stability[stability["run_id"] == "a"].iloc[0]
    row_c = stability[stability["run_id"] == "c"].iloc[0]
    assert row_a["mean_top_k_jaccard"] == pytest.approx(0.6)
    assert row_c["comparison_count"] == 2


def test_selection_baselines_do_not_require_test_for_selection():
    frame = pd.DataFrame(
        {
            "run_id": ["a", "b"],
            "validation_accuracy": [0.8, 0.7],
            "test_accuracy": [0.1, 0.99],
            "train_loss": [0.3, 0.4],
            "mean_top_k_jaccard": [0.5, 0.9],
            "median_top_k_jaccard": [0.5, 0.9],
            "is_pareto_efficient": [True, True],
        }
    )

    baselines = selection_baselines(frame)

    best_val = baselines[baselines["selection_rule"] == "best_validation_accuracy"].iloc[0]
    assert best_val["run_id"] == "a"
    assert "test_accuracy_report_only" in baselines.columns


def test_build_fairness_pareto_minimizes_gap_without_test_selection(tmp_path):
    table = pd.DataFrame(
        {
            "run_id": ["high_val", "balanced", "dominated"],
            "validation_accuracy": [0.90, 0.86, 0.84],
            "test_accuracy_report_only": [0.10, 0.99, 0.95],
            "validation_demographic_parity_gap": [0.20, 0.05, 0.10],
            "test_demographic_parity_gap_report_only": [0.01, 0.90, 0.02],
            "validation_equalized_odds_gap": [0.30, 0.04, 0.20],
            "test_equalized_odds_gap_report_only": [0.01, 0.90, 0.02],
            "validation_degree_disparity": [0.02, 0.03, 0.04],
            "test_degree_disparity_report_only": [0.90, 0.01, 0.02],
        }
    )
    table_path = tmp_path / "fairness_models.csv"
    output_prefix = tmp_path / "pareto"
    table.to_csv(table_path, index=False)

    build_fairness_pareto(table_path, output_prefix, "demographic_parity_gap")

    models = pd.read_csv(f"{output_prefix}.models.csv")
    baselines = pd.read_csv(f"{output_prefix}.baselines.csv")
    summary = json.loads((tmp_path / "pareto.summary.json").read_text(encoding="utf-8"))

    efficient = set(models.loc[models["is_pareto_efficient"], "run_id"])
    assert efficient == {"high_val", "balanced"}
    assert summary["selection_uses_test_metrics"] is False
    assert summary["test_accuracy_is_report_only"] is True
    assert summary["objective_y"] == "validation_demographic_parity_gap"

    best_val = baselines[baselines["selection_rule"] == "best_validation_accuracy"].iloc[0]
    best_gap = baselines[baselines["selection_rule"] == "best_demographic_parity_gap"].iloc[0]
    assert best_val["run_id"] == "high_val"
    assert best_gap["run_id"] == "balanced"
    assert "test_accuracy_report_only" in baselines.columns
    assert "test_demographic_parity_gap_report_only" in baselines.columns
