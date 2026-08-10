from __future__ import annotations

import numpy as np


def variation_ratio(probabilities: np.ndarray) -> np.ndarray:
    if probabilities.ndim != 3:
        raise ValueError("Expected probabilities with shape (models, nodes, classes).")
    predicted_classes = probabilities.argmax(axis=-1)
    num_models, num_nodes = predicted_classes.shape
    scores = np.zeros(num_nodes, dtype=float)
    for node_idx in range(num_nodes):
        counts = np.bincount(predicted_classes[:, node_idx])
        scores[node_idx] = 1.0 - (counts.max() / num_models)
    return scores
