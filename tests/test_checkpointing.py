import importlib.util

import pytest


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="requires torch")
def test_checkpoint_roundtrip_reproduces_logits(tmp_path):
    import torch

    from gnn_rashomon.training.checkpointing import checkpoint_logits
    from gnn_rashomon.training.trainer import build_model

    model_config = {"architecture": "mlp", "hidden_channels": 4, "dropout": 0.0}
    model = build_model(model_config, in_channels=3, out_channels=2)
    model.eval()
    x = torch.randn(5, 3)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    with torch.no_grad():
        expected = model(x, edge_index)

    checkpoint_path = tmp_path / "model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_config": model_config,
            "training_config": {},
            "metadata": {},
        },
        checkpoint_path,
    )
    actual = checkpoint_logits(checkpoint_path, x, edge_index, out_channels=2)
    assert torch.allclose(actual, expected)
