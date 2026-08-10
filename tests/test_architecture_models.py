import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")


@pytest.mark.parametrize(
    ("architecture", "extra_config"),
    [
        ("gat", {"heads": 2}),
        ("graphsage", {}),
        ("appnp", {"k": 2, "alpha": 0.1}),
    ],
)
def test_message_passing_architectures_forward(architecture, extra_config):
    from gnn_rashomon.training.trainer import build_model

    model_config = {
        "architecture": architecture,
        "hidden_channels": 4,
        "dropout": 0.0,
        **extra_config,
    }
    model = build_model(model_config, in_channels=3, out_channels=2)
    model.eval()

    x = torch.randn(5, 3)
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 1],
            [1, 2, 3, 4, 0, 0],
        ],
        dtype=torch.long,
    )

    with torch.no_grad():
        logits = model(x, edge_index)

    assert logits.shape == (5, 2)
