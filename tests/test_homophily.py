import numpy as np
import pytest

from gnn_rashomon.structure.homophily import local_homophily
from gnn_rashomon.structure.neighborhood_entropy import neighborhood_label_entropy


def test_local_homophily_handles_isolated_nodes():
    edge_index = np.array([[0, 0, 1], [1, 2, 0]])
    labels = np.array([1, 1, 0, 1])
    values = local_homophily(edge_index, labels, num_nodes=4)
    assert np.allclose(values[:2], [0.5, 1.0])
    assert np.isnan(values[2:]).all()


def test_label_structure_metrics_ignore_unlabeled_nodes():
    edge_index = np.array([[0, 0, 1, 2], [1, 2, 0, 0]])
    labels = np.array([1, -1, 0])
    label_mask = labels >= 0

    homophily = local_homophily(edge_index, labels, num_nodes=3, label_mask=label_mask)
    entropy = neighborhood_label_entropy(
        edge_index,
        labels,
        num_nodes=3,
        num_classes=2,
        label_mask=label_mask,
    )

    assert np.allclose(homophily[[0, 2]], [0.0, 0.0])
    assert np.isnan(homophily[1])
    assert np.allclose(entropy[[0, 2]], [0.0, 0.0])
    assert np.isnan(entropy[1])


def test_vectorized_entropy_handles_multiple_neighbor_classes():
    edge_index = np.array([[0, 0, 0, 1], [1, 2, 3, 0]])
    labels = np.array([0, 0, 1, 1])

    entropy = neighborhood_label_entropy(edge_index, labels, num_nodes=4, num_classes=2)

    expected = -(1 / 3) * np.log(1 / 3) - (2 / 3) * np.log(2 / 3)
    assert entropy[0] == pytest.approx(expected)
