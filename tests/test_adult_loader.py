import importlib.util

import pytest


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("torch_geometric") is None,
    reason="Adult loader test requires torch and torch-geometric.",
)
def test_adult_loader_from_csv(tmp_path):
    from gnn_rashomon.data.loaders import load_graph

    root = tmp_path / "Adult"
    root.mkdir()
    (root / "adult.csv").write_text(
        "\n".join(
            [
                "age,workclass,education,education_num,marital_status,occupation,relationship,race,sex,capital_gain,capital_loss,hours_per_week,native_country,income",
                "39,Private,Bachelors,13,Never-married,Adm-clerical,Not-in-family,White,Male,2174,0,40,United-States,<=50K",
                "50,Self-emp,Bachelors,13,Married,Exec-managerial,Husband,White,Male,0,0,13,United-States,<=50K",
                "38,Private,HS-grad,9,Divorced,Handlers-cleaners,Not-in-family,White,Female,0,0,40,United-States,<=50K",
                "53,Private,11th,7,Married,Handlers-cleaners,Husband,Black,Male,0,0,40,United-States,<=50K",
                "28,Private,Bachelors,13,Married,Prof-specialty,Wife,Black,Female,0,0,40,Cuba,>50K",
                "37,Private,Masters,14,Married,Exec-managerial,Wife,White,Female,0,0,40,United-States,>50K",
            ]
        ),
        encoding="utf-8",
    )
    bundle = load_graph(
        "adult",
        root=str(root),
        dataset_config={"source": "local", "k_neighbors": 2, "use_cache": False},
    )
    assert bundle.metadata.dataset == "adult"
    assert bundle.metadata.num_nodes == 6
    assert bundle.metadata.num_classes == 2
    assert "sex" in bundle.sensitive_attributes
    assert int(bundle.data.y.sum()) == 2
    assert all(not name.startswith("sex_") for name in bundle.data.feature_names)


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None
    or importlib.util.find_spec("torch_geometric") is None,
    reason="Adult loader test requires torch and torch-geometric.",
)
def test_adult_loader_invalidates_incompatible_cache(tmp_path):
    from gnn_rashomon.data.loaders import load_graph

    root = tmp_path / "Adult"
    root.mkdir()
    (root / "adult.csv").write_text(
        "age,sex,income\n39,Male,<=50K\n28,Female,>50K\n50,Male,<=50K\n",
        encoding="utf-8",
    )
    common = {
        "source": "local",
        "k_neighbors": 1,
        "use_cache": True,
        "train_fraction": 0.34,
        "val_fraction": 0.33,
    }
    included = load_graph(
        "adult",
        root=str(root),
        dataset_config={**common, "include_sensitive_feature": True},
    )
    excluded = load_graph(
        "adult",
        root=str(root),
        dataset_config={**common, "include_sensitive_feature": False},
    )

    assert any(name.startswith("sex_") for name in included.data.feature_names)
    assert all(not name.startswith("sex_") for name in excluded.data.feature_names)
    assert excluded.data.num_features < included.data.num_features
