from __future__ import annotations

from pathlib import Path
from typing import Any

from gnn_rashomon.training.trainer import build_model


def load_checkpoint_model(
    checkpoint_path: str | Path,
    in_channels: int,
    out_channels: int,
    map_location: str = "cpu",
) -> Any:
    import torch

    checkpoint_path = Path(checkpoint_path)
    print(f"before torch.load checkpoint {checkpoint_path}", flush=True)
    payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
    print(f"after torch.load checkpoint {checkpoint_path}", flush=True)
    model = build_model(
        model_config=payload["model_config"],
        in_channels=in_channels,
        out_channels=out_channels,
    )
    model.load_state_dict(payload["model_state_dict"])
    model = model.to(map_location)
    model.eval()
    return model


def checkpoint_logits(
    checkpoint_path: str | Path,
    x: object,
    edge_index: object,
    out_channels: int,
    map_location: str = "cpu",
) -> object:
    import torch

    model = load_checkpoint_model(
        checkpoint_path,
        in_channels=int(x.shape[1]),
        out_channels=out_channels,
        map_location=map_location,
    )
    with torch.no_grad():
        return model(x, edge_index)
