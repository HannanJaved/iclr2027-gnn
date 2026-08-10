import numpy as np

from gnn_rashomon.fairness.metrics import fairness_metrics


def test_split_fairness_ignores_predictions_outside_mask():
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    sensitive = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    degree = np.arange(1, 9)
    validation_mask = np.array([True, True, True, True, False, False, False, False])
    first = np.array([0, 1, 1, 1, 0, 0, 0, 0])
    second = np.array([0, 1, 1, 1, 1, 1, 1, 1])

    first_metrics = fairness_metrics(
        first, labels, sensitive, degree, validation_mask, degree_quantile=0.25
    )
    second_metrics = fairness_metrics(
        second, labels, sensitive, degree, validation_mask, degree_quantile=0.25
    )

    assert first_metrics == second_metrics


def test_split_fairness_rejects_empty_labeled_mask():
    values = np.array([0, 1])

    try:
        fairness_metrics(values, values, values, values, np.array([False, False]))
    except ValueError as exc:
        assert "no labeled nodes" in str(exc)
    else:
        raise AssertionError("Expected an empty fairness split to fail.")
