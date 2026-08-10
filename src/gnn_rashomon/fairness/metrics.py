from __future__ import annotations

import numpy as np


def demographic_parity_gap(
    predicted: np.ndarray,
    sensitive: np.ndarray,
    positive_label: int = 1,
) -> float:
    groups = sorted(np.unique(sensitive))
    if len(groups) != 2:
        return float("nan")
    rates = []
    for group in groups:
        mask = sensitive == group
        rates.append(
            float(np.mean(predicted[mask] == positive_label)) if np.any(mask) else float("nan")
        )
    return float(abs(rates[0] - rates[1]))


def _rate(mask: np.ndarray, values: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return float(np.mean(values[mask]))


def equalized_odds_gap(
    predicted: np.ndarray,
    labels: np.ndarray,
    sensitive: np.ndarray,
    positive_label: int = 1,
    combine: str = "max",
) -> float:
    groups = sorted(np.unique(sensitive))
    if len(groups) != 2:
        return float("nan")
    pred_pos = predicted == positive_label
    true_pos = labels == positive_label
    true_neg = ~true_pos
    tpr = [_rate((sensitive == group) & true_pos, pred_pos) for group in groups]
    fpr = [_rate((sensitive == group) & true_neg, pred_pos) for group in groups]
    gaps = np.asarray([abs(tpr[0] - tpr[1]), abs(fpr[0] - fpr[1])], dtype=float)
    if combine == "sum":
        return float(np.nansum(gaps))
    if combine == "average":
        return float(np.nanmean(gaps))
    if combine == "max":
        return float(np.nanmax(gaps))
    raise ValueError(f"Unknown equalized odds combiner: {combine}")


def degree_disparity(
    predicted: np.ndarray,
    labels: np.ndarray,
    degree: np.ndarray,
    quantile: float = 0.25,
) -> float:
    low_threshold = np.quantile(degree, quantile)
    high_threshold = np.quantile(degree, 1.0 - quantile)
    low = degree <= low_threshold
    high = degree >= high_threshold
    low_acc = _rate(low, predicted == labels)
    high_acc = _rate(high, predicted == labels)
    return float(abs(low_acc - high_acc))


def fairness_metrics(
    predicted: np.ndarray,
    labels: np.ndarray,
    sensitive: np.ndarray,
    degree: np.ndarray,
    mask: np.ndarray,
    *,
    positive_label: int = 1,
    equalized_odds_combine: str = "max",
    degree_quantile: float = 0.25,
) -> dict[str, float]:
    """Evaluate fairness metrics on exactly the supplied split mask."""
    mask = np.asarray(mask, dtype=bool) & (np.asarray(labels) >= 0)
    if not np.any(mask):
        raise ValueError("Fairness evaluation mask contains no labeled nodes.")
    return {
        "demographic_parity_gap": demographic_parity_gap(
            predicted[mask],
            sensitive[mask],
            positive_label=positive_label,
        ),
        "equalized_odds_gap": equalized_odds_gap(
            predicted[mask],
            labels[mask],
            sensitive[mask],
            positive_label=positive_label,
            combine=equalized_odds_combine,
        ),
        "degree_disparity": degree_disparity(
            predicted[mask],
            labels[mask],
            degree[mask],
            quantile=degree_quantile,
        ),
    }


def dispersion(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if values.size == 0:
        return {
            "mean": float("nan"),
            "variance": float("nan"),
            "std": float("nan"),
            "range": float("nan"),
            "iqr": float("nan"),
        }
    return {
        "mean": float(np.mean(values)),
        "variance": float(np.var(values, ddof=1)) if values.size > 1 else 0.0,
        "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
        "range": float(np.max(values) - np.min(values)),
        "iqr": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
    }
