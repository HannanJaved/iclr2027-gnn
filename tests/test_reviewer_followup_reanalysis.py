from __future__ import annotations

import numpy as np
import pandas as pd

from gnn_rashomon.analysis.counterfactual_retraining import (
    CONDITIONS,
    decode_array_task,
    expected_job_count,
    graph_path_for,
)
from gnn_rashomon.analysis.equivalence import tost_from_coefficient
from gnn_rashomon.analysis.performance_sensitivity import (
    constrain_by_validation_accuracy_only,
    train_loss_only_view,
)
from gnn_rashomon.analysis.rewiring_effect_sizes import (
    dose_response_table,
    enrich_realization_effect_sizes,
    relative_increase,
    summarize_node_delta_distribution,
)


def test_relative_increase() -> None:
    assert np.isclose(relative_increase(0.11, 0.10), 0.1)
    assert np.isnan(relative_increase(0.1, 0.0))


def test_effect_size_enrichment_and_dose_response() -> None:
    frame = pd.DataFrame(
        {
            "dataset": ["cora", "cora", "cora", "cora"],
            "mode": ["random", "random", "random", "random"],
            "strength": [0.01, 0.01, 0.25, 0.25],
            "target_homophily": [None] * 4,
            "tau": [None] * 4,
            "valid_for_inference": [True] * 4,
            "original_mean_probability_diameter": [0.10, 0.10, 0.10, 0.10],
            "rewired_mean_probability_diameter": [0.11, 0.12, 0.15, 0.14],
            "delta_mean_probability_diameter": [0.01, 0.02, 0.05, 0.04],
            "original_disagreement_fraction": [0.2, 0.2, 0.2, 0.2],
            "rewired_disagreement_fraction": [0.22, 0.24, 0.30, 0.28],
            "delta_disagreement_fraction": [0.02, 0.04, 0.10, 0.08],
            "matched_treatment_control_delta_diameter": [np.nan] * 4,
            "delta_treatment_entropy_mean": [np.nan] * 4,
        }
    )
    enriched = enrich_realization_effect_sizes(frame)
    assert np.isclose(enriched.loc[0, "relative_delta_mean_probability_diameter"], 0.1)
    dose = dose_response_table(enriched)
    assert list(dose["strength"]) == [0.01, 0.25]
    assert dose.loc[dose["strength"] == 0.25, "mean_delta_D"].iloc[0] > dose.loc[
        dose["strength"] == 0.01, "mean_delta_D"
    ].iloc[0]


def test_node_delta_distribution_thresholds() -> None:
    table = pd.DataFrame({"delta_probability_diameter": [0.0, 0.02, 0.06, 0.12, -0.01]})
    summary = summarize_node_delta_distribution(table)
    assert summary["fraction_delta_D_i_positive"] == 0.6
    assert summary["fraction_delta_D_i_gt_0p01"] == 0.6
    assert summary["fraction_delta_D_i_gt_0p05"] == 0.4
    assert summary["fraction_delta_D_i_gt_0p1"] == 0.2


def test_tost_detects_near_zero_effect() -> None:
    result = tost_from_coefficient(
        0.01,
        0.02,
        equivalence_margin=0.10,
        reference_df=40,
    )
    assert result["equivalent_at_0_05"] is True
    assert result["ci_inside_equivalence_interval"] is True


def test_tost_rejects_large_uncertain_effect() -> None:
    result = tost_from_coefficient(
        0.35,
        0.65,
        equivalence_margin=0.10,
        reference_df=17,
    )
    assert result["equivalent_at_0_05"] is False


def test_validation_only_constraint() -> None:
    source = {
        "set_id": "toy-gcn",
        "dataset": "toy",
        "architecture": "gcn",
        "candidate_run_ids": ["a", "b", "c", "d"],
        "retained_run_ids": ["a", "b"],
        "runs": {
            "a": {"validation_accuracy": 0.80, "test_accuracy": 0.70},
            "b": {"validation_accuracy": 0.79, "test_accuracy": 0.71},
            "c": {"validation_accuracy": 0.78, "test_accuracy": 0.72},
            "d": {"validation_accuracy": 0.50, "test_accuracy": 0.40},
        },
    }
    payload, summary = constrain_by_validation_accuracy_only(source, delta=0.02)
    assert payload["retained_run_ids"] == ["a", "b", "c"]
    assert "d" in payload["rejected_run_ids"]
    assert summary["membership_definition"] == "validation_accuracy_only"
    train_payload, train_summary = train_loss_only_view(source)
    assert train_payload["retained_run_ids"] == ["a", "b"]
    assert train_summary["membership_definition"] == "train_loss_only"


def test_cf_retrain_array_decoding_covers_full_grid() -> None:
    assert expected_job_count() == 2250
    dataset, condition, graph_seed, train_seed = decode_array_task(0)
    assert dataset == "cora"
    assert condition.condition_id == CONDITIONS[0].condition_id
    assert graph_seed == 0
    assert train_seed == 0
    dataset, condition, graph_seed, train_seed = decode_array_task(2249)
    assert dataset == "pubmed"
    assert condition.condition_id == CONDITIONS[-1].condition_id
    assert graph_seed == 4
    assert train_seed == 49
    path = graph_path_for("cora", CONDITIONS[0], 0)
    assert path.as_posix().endswith("cora-random-0p05-seed0.pt")
