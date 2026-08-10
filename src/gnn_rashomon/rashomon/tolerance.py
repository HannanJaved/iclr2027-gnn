from __future__ import annotations


def loss_threshold(baseline_loss: float, mode: str, epsilon: float) -> float:
    if mode == "absolute":
        return baseline_loss + epsilon
    if mode == "relative":
        return (1.0 + epsilon) * baseline_loss
    raise ValueError(f"Unknown tolerance mode: {mode}")


def is_retained(loss: float, baseline_loss: float, mode: str, epsilon: float) -> bool:
    return loss <= loss_threshold(baseline_loss, mode, epsilon)
