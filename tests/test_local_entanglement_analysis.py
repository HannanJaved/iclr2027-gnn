import pandas as pd
import pytest

from gnn_rashomon.cli.analyze_local_entanglement import (
    build_treatment_control_table,
    fit_regressions,
    run_tests,
    summarize_groups,
)


def _structure(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "node_id",
            "neighborhood_label_entropy",
            "local_homophily",
            "confidence",
            "correct",
            "degree",
        ],
    )


def test_treatment_control_table_computes_matched_local_deltas():
    metadata = {"treatment_nodes": [10, 11], "control_nodes": [20, 21]}
    original = _structure(
        [
            (10, 0.1, 0.9, 0.8, 1, 3),
            (11, 0.2, 0.8, 0.7, 1, 4),
            (20, 0.3, 0.7, 0.6, 0, 3),
            (21, 0.4, 0.6, 0.5, 0, 4),
        ]
    )
    rewired = _structure(
        [
            (10, 0.4, 0.6, 0.7, 1, 3),
            (11, 0.5, 0.5, 0.6, 0, 4),
            (20, 0.3, 0.7, 0.6, 0, 3),
            (21, 0.4, 0.6, 0.5, 0, 4),
        ]
    )
    delta = pd.DataFrame(
        {
            "original_node_id": [10, 11, 20, 21],
            "rewired_node_id": [10, 11, 20, 21],
            "delta_rashomon_capacity": [0.2, 0.4, 0.0, 0.1],
            "delta_predictive_entropy": [0.3, 0.2, 0.0, 0.0],
            "delta_variation_ratio": [0.1, 0.1, 0.0, 0.0],
        }
    )

    table = build_treatment_control_table(metadata, original, rewired, delta)

    assert set(table["group"]) == {"treatment", "control"}
    treatment = table[table["group"] == "treatment"]
    control = table[table["group"] == "control"]
    assert treatment["delta_neighborhood_label_entropy"].tolist() == pytest.approx([0.3, 0.3])
    assert control["delta_neighborhood_label_entropy"].tolist() == pytest.approx([0.0, 0.0])
    assert treatment["matched_node_id"].tolist() == [20, 21]
    assert control["matched_node_id"].tolist() == [10, 11]


def test_treatment_control_summary_and_tests_include_matched_effect():
    metadata = {"treatment_nodes": [10, 11], "control_nodes": [20, 21]}
    original = _structure(
        [
            (10, 0.1, 0.9, 0.8, 1, 3),
            (11, 0.2, 0.8, 0.7, 1, 4),
            (20, 0.3, 0.7, 0.6, 0, 3),
            (21, 0.4, 0.6, 0.5, 0, 4),
        ]
    )
    rewired = _structure(
        [
            (10, 0.3, 0.6, 0.7, 1, 3),
            (11, 0.4, 0.5, 0.6, 0, 4),
            (20, 0.3, 0.7, 0.6, 0, 3),
            (21, 0.4, 0.6, 0.5, 0, 4),
        ]
    )
    delta = pd.DataFrame(
        {
            "original_node_id": [10, 11, 20, 21],
            "rewired_node_id": [10, 11, 20, 21],
            "delta_rashomon_capacity": [0.4, 0.2, 0.0, 0.0],
            "delta_predictive_entropy": [0.3, 0.2, 0.0, 0.0],
            "delta_variation_ratio": [0.1, 0.1, 0.0, 0.0],
        }
    )
    table = build_treatment_control_table(metadata, original, rewired, delta)

    summary = summarize_groups(table, bootstrap_samples=10, seed=0)
    tests = run_tests(table, bootstrap_samples=10, seed=0)

    capacity_summary = summary[
        (summary["metric"] == "delta_rashomon_capacity") & (summary["group"] == "treatment")
    ].iloc[0]
    matched_effect = tests[
        (tests["metric"] == "delta_rashomon_capacity")
        & (tests["comparison"] == "treatment_minus_matched_control")
    ].iloc[0]

    assert capacity_summary["mean"] == pytest.approx(0.3)
    assert matched_effect["effect_mean"] == pytest.approx(0.3)
    assert "treatment_vs_control_unpaired" not in set(tests["comparison"])


def test_regressions_use_pair_fixed_effects_and_clustered_errors():
    rows = []
    for pair_id in range(20):
        for treatment_indicator in (0, 1):
            entropy_delta = 0.02 * pair_id + 0.1 * treatment_indicator
            rows.append(
                {
                    "pair_id": pair_id,
                    "treatment_indicator": treatment_indicator,
                    "delta_neighborhood_label_entropy": entropy_delta,
                    "original_degree": 2 + pair_id % 5,
                    "original_confidence": 0.5 + 0.01 * pair_id,
                    "original_correct": pair_id % 2,
                    "delta_rashomon_capacity": 0.2 * entropy_delta + 0.001 * pair_id,
                    "delta_predictive_entropy": 0.1 * entropy_delta - 0.001 * pair_id,
                }
            )

    regressions = fit_regressions(pd.DataFrame(rows))

    assert not regressions["term"].str.startswith("C(pair_id)").any()
    assert regressions["cluster_count"].eq(20).all()
    assert regressions["reference_df"].eq(19).all()
    assert regressions["note"].str.contains("fixed effects").all()
    assert regressions["note"].str.contains("pair-clustered").all()
    assert regressions["note"].str.contains("small-sample t reference").all()
