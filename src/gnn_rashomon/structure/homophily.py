from __future__ import annotations

import numpy as np


def local_homophily(
    edge_index: np.ndarray,
    labels: np.ndarray,
    num_nodes: int,
    label_mask: np.ndarray | None = None,
) -> np.ndarray:
    if label_mask is None:
        label_mask = labels >= 0
    else:
        label_mask = np.asarray(label_mask, dtype=bool) & (labels >= 0)
    src = edge_index[0].astype(np.int64, copy=False)
    dst = edge_index[1].astype(np.int64, copy=False)
    valid = label_mask[src] & label_mask[dst]
    valid_src = src[valid]
    degree = np.bincount(valid_src, minlength=num_nodes).astype(float)
    same = np.bincount(
        valid_src,
        weights=(labels[src[valid]] == labels[dst[valid]]).astype(float),
        minlength=num_nodes,
    )
    out = np.full(num_nodes, np.nan, dtype=float)
    np.divide(same, degree, out=out, where=degree > 0)
    return out
