import pandas as pd

from gnn_rashomon.cli.compare_fairgnn_baseline import _comparison_rows


def test_comparison_reports_accuracy_prioritized_deployable_choice():
    baseline = pd.DataFrame(
        {
            "baseline": ["fairsin", "fairsin"],
            "run_id": ["base-a", "base-b"],
            "validation_accuracy": [0.80, 0.86],
            "test_accuracy_report_only": [0.70, 0.72],
            "validation_demographic_parity_gap": [0.12, 0.30],
            "test_demographic_parity_gap_report_only": [0.99, 0.01],
        }
    )
    pareto = pd.DataFrame(
        {
            "run_id": ["accurate", "fair", "dominated"],
            "validation_accuracy": [0.85, 0.82, 0.70],
            "test_accuracy_report_only": [0.10, 0.99, 1.00],
            "validation_demographic_parity_gap": [0.10, 0.02, 0.40],
            "test_demographic_parity_gap_report_only": [0.90, 0.80, 0.00],
            "is_pareto_efficient": [True, True, False],
        }
    )

    rows = _comparison_rows(
        baseline,
        pareto,
        set_id="nba-gcn-seed",
        metric="demographic_parity_gap",
    )

    assert {row["selected_pareto_run_id"] for row in rows} == {"accurate"}
    assert rows[0]["selected_jointly_dominates_baseline"] is True
    assert rows[1]["selected_accuracy_beats_baseline"] is False
    assert rows[1]["selected_fairness_beats_baseline"] is True
    assert rows[1]["selected_jointly_dominates_baseline"] is False
