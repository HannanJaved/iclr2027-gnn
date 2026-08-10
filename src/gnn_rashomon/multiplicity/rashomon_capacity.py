from __future__ import annotations

import numpy as np


def _validate_probabilities(probabilities: np.ndarray) -> None:
    if probabilities.ndim != 3:
        raise ValueError("Expected probabilities with shape (models, nodes, classes).")


def pairwise_total_variation_summaries(
    probabilities: np.ndarray,
    *,
    quantile: float = 0.95,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return node-level maximum, mean, and quantile pairwise TV distances."""
    _validate_probabilities(probabilities)
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("quantile must lie in [0, 1].")
    num_models, num_nodes, _ = probabilities.shape
    if num_models < 2:
        zeros = np.zeros(num_nodes, dtype=float)
        return zeros.copy(), zeros.copy(), zeros.copy()

    pair_count = num_models * (num_models - 1) // 2
    distances = np.empty((pair_count, num_nodes), dtype=float)
    pair_index = 0
    for first in range(num_models - 1):
        count = num_models - first - 1
        distances[pair_index : pair_index + count] = 0.5 * np.abs(
            probabilities[first + 1 :] - probabilities[first]
        ).sum(axis=-1)
        pair_index += count
    return (
        distances.max(axis=0),
        distances.mean(axis=0),
        np.quantile(distances, quantile, axis=0),
    )


def _exact_capacity(probabilities: np.ndarray) -> np.ndarray:
    num_models, num_nodes, _ = probabilities.shape
    capacities = np.zeros(num_nodes, dtype=float)
    for first in range(num_models):
        distances = 0.5 * np.abs(probabilities[first + 1 :] - probabilities[first]).sum(axis=-1)
        if distances.size:
            capacities = np.maximum(capacities, distances.max(axis=0))
    return capacities


def _chunked_capacity(probabilities: np.ndarray, chunk_size: int) -> np.ndarray:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    num_models, num_nodes, _ = probabilities.shape
    capacities = np.zeros(num_nodes, dtype=float)
    for first in range(num_models):
        reference = probabilities[first : first + 1]
        for start in range(first + 1, num_models, chunk_size):
            stop = min(start + chunk_size, num_models)
            distances = 0.5 * np.abs(probabilities[start:stop] - reference).sum(axis=-1)
            if distances.size:
                capacities = np.maximum(capacities, distances.max(axis=0))
    return capacities


def _approximate_capacity(
    probabilities: np.ndarray,
    sample_pairs: int,
    seed: int,
) -> np.ndarray:
    if sample_pairs <= 0:
        raise ValueError("sample_pairs must be positive")
    num_models, num_nodes, _ = probabilities.shape
    capacities = np.zeros(num_nodes, dtype=float)
    if num_models < 2:
        return capacities
    rng = np.random.default_rng(seed)
    for _ in range(sample_pairs):
        first, second = rng.choice(num_models, size=2, replace=False)
        distances = 0.5 * np.abs(probabilities[first] - probabilities[second]).sum(axis=-1)
        capacities = np.maximum(capacities, distances)
    return capacities


def probability_diameter(
    probabilities: np.ndarray,
    method: str = "exact",
    chunk_size: int = 16,
    sample_pairs: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    _validate_probabilities(probabilities)
    if method == "exact":
        return _exact_capacity(probabilities)
    if method == "chunked":
        return _chunked_capacity(probabilities, chunk_size=chunk_size)
    if method == "approximate":
        return _approximate_capacity(
            probabilities,
            sample_pairs=sample_pairs,
            seed=seed,
        )
    raise ValueError(f"Unknown probability diameter method: {method}")


def rashomon_capacity(
    probabilities: np.ndarray,
    method: str = "exact",
    chunk_size: int = 16,
    sample_pairs: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """Backward-compatible alias for the total-variation probability diameter."""
    return probability_diameter(
        probabilities,
        method=method,
        chunk_size=chunk_size,
        sample_pairs=sample_pairs,
        seed=seed,
    )
