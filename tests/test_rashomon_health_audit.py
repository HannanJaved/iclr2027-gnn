import numpy as np
import pytest

from gnn_rashomon.cli.audit_rashomon_health import audit_model_runs, audit_tolerances


def test_health_audit_flags_collapsed_models_and_tolerance_growth():
    probabilities = {
        "good-a": np.array([[[0.8, 0.2], [0.2, 0.8], [0.7, 0.3]]]),
        "good-b": np.array([[[0.7, 0.3], [0.6, 0.4], [0.4, 0.6]]]),
        "collapsed": np.array([[[0.9, 0.1], [0.9, 0.1], [0.9, 0.1]]]),
    }
    runs = [
        {
            "run_id": "a",
            "train_loss": 1.0,
            "validation_accuracy": 0.8,
            "test_accuracy": 0.7,
            "probability_path": "good-a",
        },
        {
            "run_id": "b",
            "train_loss": 1.05,
            "validation_accuracy": 0.75,
            "test_accuracy": 0.65,
            "probability_path": "good-b",
        },
        {
            "run_id": "c",
            "train_loss": 1.8,
            "validation_accuracy": 0.5,
            "test_accuracy": 0.5,
            "probability_path": "collapsed",
        },
    ]

    table = audit_tolerances(
        runs=runs,
        epsilons=[0.1, 1.0],
        mode="relative",
        collapse_majority_threshold=0.99,
        probability_loader=lambda path: probabilities[path][0],
    )

    strict, broad = table.iloc[0], table.iloc[1]
    assert strict["retained_count"] == 2
    assert strict["collapsed_model_count"] == 0
    assert broad["retained_count"] == 3
    assert broad["collapsed_model_count"] == 1
    assert broad["collapsed_model_fraction"] == pytest.approx(1 / 3)
    assert broad["fraction_prediction_disagreement"] == pytest.approx(2 / 3)

    models = audit_model_runs(
        runs=runs,
        collapse_majority_threshold=0.99,
        probability_loader=lambda path: probabilities[path][0],
    )
    collapsed = models.set_index("run_id")["collapsed_prediction"].to_dict()
    assert collapsed == {"a": False, "b": False, "c": True}
