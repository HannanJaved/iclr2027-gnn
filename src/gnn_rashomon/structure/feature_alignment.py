from __future__ import annotations

import numpy as np


def node_feature_similarity(edge_index: np.ndarray, features: np.ndarray, num_nodes: int) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1)
    scores = np.zeros(num_nodes, dtype=float)
    degree = np.zeros(num_nodes, dtype=float)
    for src, dst in edge_index.T:
        denominator = norms[src] * norms[dst]
        similarity = 0.0 if denominator == 0.0 else float(np.dot(features[src], features[dst]) / denominator)
        scores[src] += similarity
        degree[src] += 1.0
    out = np.zeros(num_nodes, dtype=float)
    np.divide(scores, degree, out=out, where=degree > 0)
    return out


def graph_feature_similarity(edge_index: np.ndarray, features: np.ndarray, num_nodes: int) -> float:
    return float(node_feature_similarity(edge_index, features, num_nodes).mean())
