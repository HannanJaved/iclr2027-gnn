from __future__ import annotations

from typing import Any


def _accuracy_summary(
    runs: dict[str, Any],
    retained: list[str],
) -> dict[str, float]:
    if not retained:
        return {
            "validation_accuracy_min": float("nan"),
            "validation_accuracy_max": float("nan"),
            "test_accuracy_min_report_only": float("nan"),
            "test_accuracy_max_report_only": float("nan"),
        }
    validation = [float(runs[run_id]["validation_accuracy"]) for run_id in retained]
    test = [float(runs[run_id]["test_accuracy"]) for run_id in retained]
    return {
        "validation_accuracy_min": min(validation),
        "validation_accuracy_max": max(validation),
        "test_accuracy_min_report_only": min(test),
        "test_accuracy_max_report_only": max(test),
    }


def constrain_by_validation_accuracy(
    rashomon: dict[str, Any],
    delta: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if delta < 0.0:
        raise ValueError("Validation-accuracy delta must be non-negative.")
    train_eligible = [str(run_id) for run_id in rashomon["retained_run_ids"]]
    if not train_eligible:
        raise ValueError("The source Rashomon set has no retained runs.")
    runs = rashomon["runs"]
    best_validation_accuracy = max(
        float(runs[run_id]["validation_accuracy"]) for run_id in train_eligible
    )
    threshold = best_validation_accuracy - delta
    retained = sorted(
        run_id
        for run_id in train_eligible
        if float(runs[run_id]["validation_accuracy"]) >= threshold - 1e-12
    )
    rejected_by_validation = sorted(set(train_eligible) - set(retained))
    delta_id = f"{delta:g}".replace(".", "p")
    set_id = f"{rashomon['set_id']}-valaccdelta{delta_id}"
    payload = dict(rashomon)
    payload.update(
        {
            "set_id": set_id,
            "parent_set_id": rashomon["set_id"],
            "membership_definition": "joint_train_loss_and_validation_accuracy",
            "train_eligible_run_ids": sorted(train_eligible),
            "validation_accuracy_delta": delta,
            "best_train_eligible_validation_accuracy": best_validation_accuracy,
            "validation_accuracy_threshold": threshold,
            "validation_constraint_definition": (
                "validation_accuracy >= best validation accuracy among train-eligible runs "
                "minus validation_accuracy_delta"
            ),
            "retained_run_ids": retained,
            "rejected_by_validation_run_ids": rejected_by_validation,
            "retained_count": len(retained),
            "rejected_count": len(rashomon["candidate_run_ids"]) - len(retained),
            "test_metrics_used_for_membership": False,
        }
    )
    summary = {
        "set_id": set_id,
        "parent_set_id": rashomon["set_id"],
        "dataset": rashomon["dataset"],
        "architecture": rashomon["architecture"],
        "membership_definition": "joint_train_loss_and_validation_accuracy",
        "validation_accuracy_delta": delta,
        "validation_accuracy_threshold": threshold,
        "train_eligible_count": len(train_eligible),
        "retained_count": len(retained),
        "test_metrics_used_for_membership": False,
        **_accuracy_summary(runs, retained),
    }
    return payload, summary


def constrain_by_validation_accuracy_only(
    rashomon: dict[str, Any],
    delta: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Filter the full candidate pool by validation accuracy alone (no train-loss gate)."""
    if delta < 0.0:
        raise ValueError("Validation-accuracy delta must be non-negative.")
    candidates = [str(run_id) for run_id in rashomon["candidate_run_ids"]]
    if not candidates:
        raise ValueError("The source Rashomon set has no candidate runs.")
    runs = rashomon["runs"]
    best_validation_accuracy = max(
        float(runs[run_id]["validation_accuracy"]) for run_id in candidates
    )
    threshold = best_validation_accuracy - delta
    retained = sorted(
        run_id
        for run_id in candidates
        if float(runs[run_id]["validation_accuracy"]) >= threshold - 1e-12
    )
    rejected = sorted(set(candidates) - set(retained))
    delta_id = f"{delta:g}".replace(".", "p")
    set_id = f"{rashomon['set_id']}-valonlydelta{delta_id}"
    payload = dict(rashomon)
    payload.update(
        {
            "set_id": set_id,
            "parent_set_id": rashomon["set_id"],
            "membership_definition": "validation_accuracy_only",
            "validation_accuracy_delta": delta,
            "best_candidate_validation_accuracy": best_validation_accuracy,
            "validation_accuracy_threshold": threshold,
            "validation_constraint_definition": (
                "validation_accuracy >= best validation accuracy among candidates "
                "minus validation_accuracy_delta"
            ),
            "retained_run_ids": retained,
            "rejected_run_ids": rejected,
            "retained_count": len(retained),
            "rejected_count": len(rejected),
            "test_metrics_used_for_membership": False,
        }
    )
    summary = {
        "set_id": set_id,
        "parent_set_id": rashomon["set_id"],
        "dataset": rashomon["dataset"],
        "architecture": rashomon["architecture"],
        "membership_definition": "validation_accuracy_only",
        "validation_accuracy_delta": delta,
        "validation_accuracy_threshold": threshold,
        "candidate_count": len(candidates),
        "retained_count": len(retained),
        "test_metrics_used_for_membership": False,
        **_accuracy_summary(runs, retained),
    }
    return payload, summary


def train_loss_only_view(rashomon: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Identity view that tags the formal train-loss Rashomon set for ablation tables."""
    retained = [str(run_id) for run_id in rashomon["retained_run_ids"]]
    payload = dict(rashomon)
    payload.update(
        {
            "parent_set_id": rashomon["set_id"],
            "membership_definition": "train_loss_only",
            "test_metrics_used_for_membership": False,
        }
    )
    summary = {
        "set_id": rashomon["set_id"],
        "parent_set_id": rashomon["set_id"],
        "dataset": rashomon["dataset"],
        "architecture": rashomon["architecture"],
        "membership_definition": "train_loss_only",
        "validation_accuracy_delta": None,
        "retained_count": len(retained),
        "test_metrics_used_for_membership": False,
        **_accuracy_summary(rashomon["runs"], retained),
    }
    return payload, summary
