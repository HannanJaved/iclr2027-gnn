import sys
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")


def test_ogbn_arxiv_loader_uses_official_split(monkeypatch, tmp_path):
    from torch_geometric.data import Data

    import gnn_rashomon.data.loaders as loaders

    class FakeDataset:
        num_classes = 3

        def __init__(self, name, root):
            assert name == "ogbn-arxiv"
            self.root = root

        def __getitem__(self, index):
            assert index == 0
            return Data(
                x=torch.ones(6, 4),
                y=torch.tensor([[0], [1], [2], [0], [1], [2]]),
                edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
            )

        def get_idx_split(self):
            return {
                "train": torch.tensor([0, 1, 2]),
                "valid": torch.tensor([3]),
                "test": torch.tensor([4, 5]),
            }

    module = types.ModuleType("ogb.nodeproppred")
    module.PygNodePropPredDataset = FakeDataset
    monkeypatch.setitem(sys.modules, "ogb", types.ModuleType("ogb"))
    monkeypatch.setitem(sys.modules, "ogb.nodeproppred", module)

    bundle = loaders.load_graph(
        "ogbn_arxiv",
        root=str(tmp_path),
        dataset_config={"to_undirected": False, "normalize_features": False},
    )

    assert bundle.metadata.dataset == "ogbn_arxiv"
    assert tuple(bundle.data.y.shape) == (6,)
    assert int(bundle.data.train_mask.sum()) == 3
    assert int(bundle.data.val_mask.sum()) == 1
    assert int(bundle.data.test_mask.sum()) == 2
    assert bundle.sensitive_attributes == {}
