from __future__ import annotations

import numpy as np


def predictive_entropy(probabilities: np.ndarray, delta: float = 1e-12) -> np.ndarray:
    if probabilities.ndim != 3:
        raise ValueError("Expected probabilities with shape (models, nodes, classes).")
    mean_probabilities = probabilities.mean(axis=0)
    return -np.sum(mean_probabilities * np.log(mean_probabilities + delta), axis=-1)
