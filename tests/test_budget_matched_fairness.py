from __future__ import annotations

import pandas as pd

from gnn_rashomon.analysis.budget_matched_fairness import (
    budget_matched_comparisons,
    summarize_budget_matched,
)


def test_budget_matching_samples_before_validation_only_selection() -> None:
    baseline = pd.DataFrame(
        {
            "baseline": ["fair", "fair"],
            "run_id": ["b1", "b2"],
            "validation_accuracy": [0.70, 0.75],
            "validation_demographic_parity_gap": [0.20, 0.20],
        }
    )
    candidates = pd.DataFrame(
        {
            "run_id": ["c1", "c2", "c3", "c4"],
            "validation_accuracy": [0.80, 0.79, 0.65, 0.60],
            "validation_demographic_parity_gap": [0.10, 0.05, 0.01, 0.00],
        }
    )

    draws = budget_matched_comparisons(
        baseline,
        candidates,
        metric="demographic_parity_gap",
        sample_size=2,
        draws=20,
        seed=11,
        set_id="candidate-set",
    )
    summary = summarize_budget_matched(draws)

    assert len(draws) == 20
    assert set(draws["sample_size"]) == {2}
    assert draws["selected_accuracy_win_rate"].between(0, 1).all()
    assert summary.loc[0, "draw_count"] == 20
