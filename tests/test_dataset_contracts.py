from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from gnn_rashomon.data.preprocessing import standardize_features
from gnn_rashomon.data.splits import deterministic_masks, validate_masks


def test_deterministic_masks_are_reproducible_and_valid() -> None:
    first = deterministic_masks(num_nodes=20, train_fraction=0.5, val_fraction=0.25, seed=17)
    second = deterministic_masks(num_nodes=20, train_fraction=0.5, val_fraction=0.25, seed=17)

    for left, right in zip(first, second):
        assert np.array_equal(left, right)

    split = validate_masks(*first, num_nodes=20)
    assert split.valid
    assert split.train_count == 10
    assert split.validation_count == 5
    assert split.test_count == 5


def test_validate_masks_rejects_overlapping_masks() -> None:
    train = np.array([True, True, False])
    val = np.array([False, True, False])
    test = np.array([False, False, True])

    split = validate_masks(train, val, test, num_nodes=3)
    assert not split.valid
    assert split.covers_all_nodes
    assert not split.mutually_exclusive


def test_validate_masks_can_allow_unassigned_public_split_nodes() -> None:
    train = np.array([True, False, False])
    val = np.array([False, True, False])
    test = np.array([False, False, False])

    strict = validate_masks(train, val, test, num_nodes=3)
    relaxed = validate_masks(train, val, test, num_nodes=3, allow_unassigned=True)

    assert strict.unassigned_count == 1
    assert not strict.valid
    assert relaxed.valid


def test_standardize_features_handles_constant_columns() -> None:
    features = np.array([[1.0, 2.0], [1.0, 4.0], [1.0, 6.0]], dtype=np.float32)
    standardized = standardize_features(features)

    assert np.allclose(standardized[:, 0], 0.0)
    assert np.isfinite(standardized).all()


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("torch_geometric") is None,
    reason="Dataset validation CLI requires torch and torch-geometric.",
)
def test_validate_dataset_cli_writes_adult_metadata(tmp_path, monkeypatch) -> None:
    from gnn_rashomon.cli.validate_dataset import main

    project = tmp_path
    configs = project / "configs"
    (configs / "datasets").mkdir(parents=True)
    (configs / "models").mkdir()
    (project / "data" / "Adult").mkdir(parents=True)

    (configs / "base.yaml").write_text(
        "\n".join(
            [
                "dataset: adult",
                "model: gcn",
                "paths:",
                "  data_dir: data",
                "  output_dir: outputs",
            ]
        ),
        encoding="utf-8",
    )
    (configs / "datasets" / "adult.yaml").write_text(
        "\n".join(
            [
                "name: adult",
                "root: data/Adult",
                "source: local",
                "k_neighbors: 2",
                "use_cache: false",
                "split_seed: 0",
                "train_fraction: 0.5",
                "val_fraction: 0.25",
            ]
        ),
        encoding="utf-8",
    )
    (configs / "models" / "gcn.yaml").write_text("architecture: gcn\n", encoding="utf-8")
    (project / "data" / "Adult" / "adult.csv").write_text(
        "\n".join(
            [
                "age,workclass,education,education_num,marital_status,occupation,relationship,race,sex,capital_gain,capital_loss,hours_per_week,native_country,income",
                "39,Private,Bachelors,13,Never-married,Adm-clerical,Not-in-family,White,Male,2174,0,40,United-States,<=50K",
                "50,Self-emp,Bachelors,13,Married,Exec-managerial,Husband,White,Male,0,0,13,United-States,<=50K",
                "38,Private,HS-grad,9,Divorced,Handlers-cleaners,Not-in-family,White,Female,0,0,40,United-States,<=50K",
                "53,Private,11th,7,Married,Handlers-cleaners,Husband,Black,Male,0,0,40,United-States,<=50K",
                "28,Private,Bachelors,13,Married,Prof-specialty,Wife,Black,Female,0,0,40,Cuba,>50K",
                "37,Private,Masters,14,Married,Exec-managerial,Wife,White,Female,0,0,40,United-States,>50K",
                "49,Private,HS-grad,9,Married,Craft-repair,Husband,White,Male,0,0,40,United-States,>50K",
                "52,Private,HS-grad,9,Married,Exec-managerial,Husband,White,Male,0,0,45,United-States,>50K",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.chdir(project)
    main([])

    report = project / "outputs" / "metadata" / "adult.metadata.json"
    assert report.exists()
    assert '"mutually_exclusive": true' in report.read_text(encoding="utf-8")
