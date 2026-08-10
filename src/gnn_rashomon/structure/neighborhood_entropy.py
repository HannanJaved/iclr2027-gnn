from __future__ import annotations

import numpy as np


def neighborhood_label_entropy(
    edge_index: np.ndarray,
    labels: np.ndarray,
    num_nodes: int,
    num_classes: int,
    label_mask: np.ndarray | None = None,
) -> np.ndarray:
    if label_mask is None:
        label_mask = labels >= 0
    else:
        label_mask = np.asarray(label_mask, dtype=bool) & (labels >= 0)
    src = edge_index[0].astype(np.int64, copy=False)
    dst = edge_index[1].astype(np.int64, copy=False)
    valid = label_mask[src] & label_mask[dst]
    flat_bins = src[valid] * num_classes + labels[dst[valid]]
    counts = np.bincount(flat_bins, minlength=num_nodes * num_classes).reshape(
        num_nodes, num_classes
    )
    totals = counts.sum(axis=1, keepdims=True)
    probabilities = np.divide(
        counts,
        totals,
        out=np.zeros_like(counts, dtype=float),
        where=totals > 0,
    )
    log_probabilities = np.zeros_like(probabilities)
    np.log(probabilities, out=log_probabilities, where=probabilities > 0)
    entropies = -np.sum(probabilities * log_probabilities, axis=1)
    entropies[(totals[:, 0] == 0) | ~label_mask] = np.nan
    return entropies
