from __future__ import annotations

import numpy as np
import pandas as pd

from gnn_rashomon.multiplicity.predictive_entropy import predictive_entropy
from gnn_rashomon.multiplicity.rashomon_capacity import pairwise_total_variation_summaries
from gnn_rashomon.multiplicity.variation_ratio import variation_ratio

METRIC_COLUMNS = [
    "disagreement_fraction",
    "mean_predictive_entropy",
    "mean_variation_ratio",
    "mean_probability_diameter",
    "mean_pairwise_total_variation",
    "mean_pairwise_total_variation_q95",
]


def multiplicity_summary(probabilities: np.ndarray) -> dict[str, float]:
    if probabilities.ndim != 3:
        raise ValueError("Expected probabilities with shape (models, nodes, classes).")
    if probabilities.shape[0] < 2:
        raise ValueError("At least two models are required for multiplicity robustness.")
    predictions = probabilities.argmax(axis=-1)
    diameter, pairwise_mean, pairwise_q95 = pairwise_total_variation_summaries(
        probabilities, quantile=0.95
    )
    return {
        "disagreement_fraction": float(
            (predictions != predictions[0:1]).any(axis=0).mean()
        ),
        "mean_predictive_entropy": float(predictive_entropy(probabilities).mean()),
        "mean_variation_ratio": float(variation_ratio(probabilities).mean()),
        "mean_probability_diameter": float(diameter.mean()),
        "mean_pairwise_total_variation": float(pairwise_mean.mean()),
        "mean_pairwise_total_variation_q95": float(pairwise_q95.mean()),
    }


def subsample_multiplicity(
    probabilities: np.ndarray,
    model_ids: list[str],
    *,
    sample_size: int,
    draws: int,
    seed: int,
) -> pd.DataFrame:
    if len(model_ids) != probabilities.shape[0]:
        raise ValueError("model_ids must match the model axis of probabilities.")
    if not 2 <= sample_size <= probabilities.shape[0]:
        raise ValueError("sample_size must be between 2 and the retained model count.")
    if draws <= 0:
        raise ValueError("draws must be positive.")

    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for draw in range(draws):
        indices = np.sort(rng.choice(probabilities.shape[0], size=sample_size, replace=False))
        rows.append(
            {
                "draw": draw,
                "sample_size": sample_size,
                "sampled_model_ids": ";".join(model_ids[index] for index in indices),
                **multiplicity_summary(probabilities[indices]),
            }
        )
    return pd.DataFrame(rows)


def summarize_subsamples(draws: pd.DataFrame) -> dict[str, dict[str, float]]:
    missing = set(METRIC_COLUMNS) - set(draws.columns)
    if missing:
        raise ValueError(f"Missing robustness metrics: {sorted(missing)}")
    output: dict[str, dict[str, float]] = {}
    for metric in METRIC_COLUMNS:
        values = draws[metric].to_numpy(dtype=float)
        low, median, high = np.quantile(values, [0.025, 0.5, 0.975])
        output[metric] = {
            "mean": float(values.mean()),
            "median": float(median),
            "interval_95_low": float(low),
            "interval_95_high": float(high),
        }
    return output
