from gnn_rashomon.analysis.performance_sensitivity import constrain_by_validation_accuracy


def test_validation_constraint_uses_only_train_eligible_runs_and_not_test_accuracy():
    source = {
        "set_id": "toy-gcn-hyper-relative2",
        "dataset": "toy",
        "architecture": "gcn",
        "candidate_run_ids": ["a", "b", "c", "d"],
        "retained_run_ids": ["a", "b", "c"],
        "runs": {
            "a": {"validation_accuracy": 0.80, "test_accuracy": 0.10},
            "b": {"validation_accuracy": 0.79, "test_accuracy": 0.99},
            "c": {"validation_accuracy": 0.75, "test_accuracy": 1.00},
            "d": {"validation_accuracy": 0.90, "test_accuracy": 1.00},
        },
    }

    constrained, summary = constrain_by_validation_accuracy(source, delta=0.02)

    assert constrained["retained_run_ids"] == ["a", "b"]
    assert constrained["rejected_by_validation_run_ids"] == ["c"]
    assert "d" not in constrained["train_eligible_run_ids"]
    assert constrained["test_metrics_used_for_membership"] is False
    assert summary["validation_accuracy_threshold"] == 0.78
    assert summary["test_accuracy_min_report_only"] == 0.10
    assert summary["test_accuracy_max_report_only"] == 0.99


def test_validation_constraint_rejects_negative_delta():
    source = {
        "set_id": "toy",
        "dataset": "toy",
        "architecture": "gcn",
        "candidate_run_ids": ["a"],
        "retained_run_ids": ["a"],
        "runs": {"a": {"validation_accuracy": 0.8, "test_accuracy": 0.8}},
    }

    try:
        constrain_by_validation_accuracy(source, delta=-0.01)
    except ValueError as exc:
        assert "non-negative" in str(exc)
    else:
        raise AssertionError("Expected negative validation delta to fail.")
