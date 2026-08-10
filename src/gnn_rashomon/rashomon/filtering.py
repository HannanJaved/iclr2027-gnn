from __future__ import annotations

from collections.abc import Iterable

from gnn_rashomon.rashomon.tolerance import is_retained


def filter_by_training_loss(
    losses: dict[str, float],
    baseline_loss: float,
    mode: str,
    epsilon: float,
) -> tuple[list[str], list[str]]:
    retained: list[str] = []
    rejected: list[str] = []
    for run_id, loss in losses.items():
        target = retained if is_retained(loss, baseline_loss, mode, epsilon) else rejected
        target.append(run_id)
    return retained, rejected


def validate_partition(candidates: Iterable[str], retained: Iterable[str], rejected: Iterable[str]) -> None:
    candidate_set = set(candidates)
    retained_set = set(retained)
    rejected_set = set(rejected)
    if retained_set & rejected_set:
        raise ValueError("Retained and rejected sets overlap.")
    if retained_set | rejected_set != candidate_set:
        raise ValueError("Retained and rejected sets do not partition candidates.")
