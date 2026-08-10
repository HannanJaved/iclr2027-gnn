from __future__ import annotations

import torch
from torch_geometric.data import Data

from gnn_rashomon.data.loaders import _load_processed_planetoid


def test_cached_planetoid_applies_runtime_feature_normalization(tmp_path) -> None:
    data = Data(
        x=torch.tensor([[1.0, 3.0], [0.0, 2.0]]),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        y=torch.tensor([0, 1], dtype=torch.long),
        train_mask=torch.tensor([True, False]),
        val_mask=torch.tensor([False, True]),
        test_mask=torch.tensor([False, True]),
    )
    path = tmp_path / "data.pt"
    torch.save(data, path)

    graph = _load_processed_planetoid("cora", path)

    assert torch.allclose(graph.data.x.sum(dim=1), torch.ones(2))
    assert torch.allclose(graph.data.x, torch.tensor([[0.25, 0.75], [0.0, 1.0]]))
