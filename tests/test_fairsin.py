import pytest

torch = pytest.importorskip("torch")

from gnn_rashomon.fairness.fairsin import heterogeneous_neighbor_targets


def test_heterogeneous_neighbor_targets_average_only_other_group_neighbors():
    features = torch.tensor([[1.0, 0.0], [0.0, 2.0], [2.0, 2.0], [4.0, 0.0]])
    sensitive = torch.tensor([0, 1, 1, 0])
    edge_index = torch.tensor(
        [[0, 0, 1, 2, 3], [1, 2, 0, 3, 2]], dtype=torch.long
    )

    targets, has_target = heterogeneous_neighbor_targets(features, edge_index, sensitive)

    assert torch.allclose(targets[0], torch.tensor([1.0, 2.0]))
    assert torch.allclose(targets[1], features[0])
    assert torch.allclose(targets[2], features[3])
    assert torch.allclose(targets[3], features[2])
    assert bool(has_target.all())
