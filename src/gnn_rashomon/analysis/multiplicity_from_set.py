from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from gnn_rashomon.multiplicity.rashomon_capacity import pairwise_total_variation_summaries
from gnn_rashomon.multiplicity.variation_ratio import variation_ratio


def load_retained_probabilities(
    rashomon: dict[str, Any],
    *,
    project_root: Path | None = None,
) -> np.ndarray:
    import torch

    retained = [str(run_id) for run_id in rashomon["retained_run_ids"]]
    if not retained:
        raise ValueError("Rashomon set has no retained runs.")
    runs = rashomon["runs"]
    arrays = []
    for run_id in retained:
        path = Path(str(runs[run_id]["probability_path"]))
        if not path.is_absolute() and project_root is not None:
            path = project_root / path
        arrays.append(torch.load(path, map_location="cpu", weights_only=True).detach().cpu().numpy())
    return np.stack(arrays)


def validation_accuracy_distribution(rashomon: dict[str, Any]) -> dict[str, float]:
    retained = [str(run_id) for run_id in rashomon["retained_run_ids"]]
    if not retained:
        return {
            "validation_accuracy_mean": float("nan"),
            "validation_accuracy_std": float("nan"),
            "validation_accuracy_min": float("nan"),
            "validation_accuracy_max": float("nan"),
        }
    values = np.asarray(
        [float(rashomon["runs"][run_id]["validation_accuracy"]) for run_id in retained],
        dtype=float,
    )
    return {
        "validation_accuracy_mean": float(values.mean()),
        "validation_accuracy_std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "validation_accuracy_min": float(values.min()),
        "validation_accuracy_max": float(values.max()),
    }


def multiplicity_metrics_from_probabilities(probabilities: np.ndarray) -> dict[str, float]:
    if probabilities.ndim != 3:
        raise ValueError("Expected probabilities with shape (models, nodes, classes).")
    predictions = probabilities.argmax(axis=-1)
    disagreement = (predictions != predictions[0:1]).any(axis=0)
    diameter, pairwise_mean, _ = pairwise_total_variation_summaries(probabilities)
    return {
        "retained_count": int(probabilities.shape[0]),
        "fraction_prediction_disagreement": float(disagreement.mean()),
        "mean_probability_diameter": float(diameter.mean()),
        "mean_pairwise_total_variation": float(pairwise_mean.mean()),
        "mean_variation_ratio": float(variation_ratio(probabilities).mean()),
    }


def multiplicity_metrics_from_set(
    rashomon: dict[str, Any],
    *,
    project_root: Path | None = None,
) -> dict[str, float]:
    probabilities = load_retained_probabilities(rashomon, project_root=project_root)
    metrics = multiplicity_metrics_from_probabilities(probabilities)
    metrics.update(validation_accuracy_distribution(rashomon))
    return metrics
