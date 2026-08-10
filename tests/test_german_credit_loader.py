import importlib.util

import pytest


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("torch_geometric") is None,
    reason="German Credit loader test requires torch and torch-geometric.",
)
def test_german_credit_loader_builds_full_small_graph(tmp_path):
    from gnn_rashomon.data.loaders import load_graph

    root = tmp_path / "GermanCredit"
    root.mkdir()
    data_path = root / "german.data"
    data_path.write_text(
        "\n".join(
            [
                "A11 6 A34 A43 1169 A65 A75 4 A93 A101 4 A121 67 A143 A152 2 A173 1 A192 A201 1",
                "A12 48 A32 A43 5951 A61 A73 2 A92 A101 2 A121 22 A143 A152 1 A173 1 A191 A201 2",
                "A14 12 A34 A46 2096 A61 A74 2 A93 A101 3 A121 49 A143 A152 1 A172 2 A191 A201 1",
                "A11 42 A32 A42 7882 A61 A74 2 A93 A103 4 A122 45 A143 A153 1 A173 2 A191 A201 1",
                "A11 24 A33 A40 4870 A61 A73 3 A93 A101 4 A124 53 A143 A153 2 A173 2 A191 A201 2",
                "A14 36 A32 A46 9055 A65 A73 2 A93 A101 4 A124 35 A143 A153 1 A172 2 A192 A201 1",
            ]
        ),
        encoding="utf-8",
    )

    bundle = load_graph(
        "german_credit",
        root=str(root),
        dataset_config={
            "data_path": str(data_path),
            "k_neighbors": 2,
            "use_cache": False,
            "include_sensitive_feature": False,
        },
    )

    assert bundle.metadata.dataset == "german_credit"
    assert bundle.metadata.num_nodes == 6
    assert bundle.metadata.num_classes == 2
    assert "sex" in bundle.sensitive_attributes
    assert "age_ge_threshold" in bundle.sensitive_attributes
    assert int(bundle.data.y.sum()) == 4
    assert int(bundle.sensitive_attributes["sex"].sum()) == 5
