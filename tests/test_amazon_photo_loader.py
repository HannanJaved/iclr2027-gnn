import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")


def test_amazon_photo_loader_adds_deterministic_masks(monkeypatch, tmp_path):
    from torch_geometric.data import Data

    import gnn_rashomon.data.loaders as loaders

    class FakeAmazon:
        num_classes = 3

        def __init__(self, root, name, transform=None):
            self.root = root
            self.name = name
            self.transform = transform

        def __getitem__(self, index):
            assert index == 0
            return Data(
                x=torch.ones(10, 4),
                y=torch.arange(10) % 3,
                edge_index=torch.tensor(
                    [
                        [0, 1, 2, 3, 4, 5, 6, 7],
                        [1, 2, 3, 4, 5, 6, 7, 8],
                    ],
                    dtype=torch.long,
                ),
            )

    monkeypatch.setattr(loaders, "Path", loaders.Path)
    monkeypatch.setattr("torch_geometric.datasets.Amazon", FakeAmazon)

    bundle = loaders.load_graph(
        "amazon_photo",
        root=str(tmp_path),
        dataset_config={
            "train_fraction": 0.5,
            "val_fraction": 0.2,
            "split_seed": 7,
        },
    )

    assert bundle.metadata.dataset == "amazon_photo"
    assert bundle.metadata.num_nodes == 10
    assert bundle.metadata.num_classes == 3
    assert int(bundle.data.train_mask.sum()) == 5
    assert int(bundle.data.val_mask.sum()) == 2
    assert int(bundle.data.test_mask.sum()) == 3
    assert bundle.sensitive_attributes == {}
