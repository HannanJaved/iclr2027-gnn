import numpy as np

from gnn_rashomon.structure.community_boundaries import boundary_features, clustering_coefficient


def test_clustering_and_boundary_features():
    edge_index = np.array(
        [
            [0, 1, 1, 2, 2, 0, 2, 3, 3, 4, 4, 2],
            [1, 0, 2, 1, 0, 2, 3, 2, 4, 3, 2, 4],
        ]
    )
    clustering = clustering_coefficient(edge_index, num_nodes=5)
    assert clustering[0] == 1.0
    features = boundary_features(edge_index, num_nodes=5, seed=0)
    assert set(features) == {
        "community_id",
        "is_community_boundary",
        "cross_community_fraction",
        "distance_to_community_boundary",
    }
    assert features["community_id"].shape == (5,)
