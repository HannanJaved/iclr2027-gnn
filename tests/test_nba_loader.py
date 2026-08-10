import importlib.util

import pytest


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("torch_geometric") is None,
    reason="NBA loader test requires torch and torch-geometric.",
)
def test_nba_loader_uses_labeled_masks_and_country_sensitive_attribute(tmp_path):
    from gnn_rashomon.data.loaders import load_graph

    root = tmp_path / "NBA"
    root.mkdir()
    csv_path = root / "nba.csv"
    relationship_path = root / "nba_relationship.txt"
    csv_path.write_text(
        "\n".join(
            [
                "user_id,SALARY,AGE,MP,POINTS,country,C,PG",
                "10,1,25,10.0,8.0,0,1,0",
                "11,0,28,20.0,12.0,1,0,1",
                "12,-1,24,15.0,4.0,0,0,1",
                "13,1,31,30.0,16.0,1,1,0",
                "14,0,22,11.0,3.0,0,0,1",
                "15,1,27,18.0,9.0,1,1,0",
            ]
        ),
        encoding="utf-8",
    )
    relationship_path.write_text(
        "\n".join(["10\t11", "11\t12", "12\t13", "13\t14", "14\t15"]),
        encoding="utf-8",
    )

    bundle = load_graph(
        "nba",
        root=str(root),
        dataset_config={
            "csv_path": str(csv_path),
            "relationship_path": str(relationship_path),
            "label_number": 2,
            "val_fraction_labeled": 0.33,
            "split_seed": 1,
            "use_cache": False,
        },
    )

    assert bundle.metadata.dataset == "nba"
    assert bundle.metadata.num_nodes == 6
    assert bundle.metadata.num_classes == 2
    assert "country" in bundle.sensitive_attributes
    assert int(bundle.data.label_mask.sum()) == 5
    assert bool(bundle.data.label_mask[2]) is False
    assert int(bundle.data.train_mask.sum()) == 2
    assert int(bundle.data.val_mask.sum()) >= 1
    assert int(bundle.data.test_mask.sum()) >= 1
