from __future__ import annotations

import numpy as np


def node_degree(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    return np.bincount(edge_index[0], minlength=num_nodes)
