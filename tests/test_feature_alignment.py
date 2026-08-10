import numpy as np

from gnn_rashomon.structure.feature_alignment import (
    graph_feature_similarity,
    node_feature_similarity,
)


def test_feature_alignment_known_values():
    edge_index = np.array([[0, 0, 1], [1, 2, 0]])
    features = np.array([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 0.0]])
    node_scores = node_feature_similarity(edge_index, features, num_nodes=4)
    assert np.allclose(node_scores, [0.5, 1.0, 0.0, 0.0])
    assert graph_feature_similarity(edge_index, features, num_nodes=4) == 0.375
