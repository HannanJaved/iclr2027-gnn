import importlib.util

import pytest


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("torch_geometric") is None,
    reason="GCN smoke test requires torch and torch-geometric.",
)
def test_train_baseline_smoke(tmp_path, monkeypatch):
    from gnn_rashomon.cli.train_baseline import main

    monkeypatch.chdir(tmp_path)
    main(["dataset=cora", "trainer.max_epochs=5", f"paths.output_dir={tmp_path / 'outputs'}"])
