from __future__ import annotations

from typing import Any


def heterogeneous_neighbor_targets(
    x: Any,
    edge_index: Any,
    sensitive: Any,
) -> tuple[Any, Any]:
    """Return mean features of outgoing neighbors from the other sensitive group."""
    import torch

    src, dst = edge_index
    keep = sensitive[src] != sensitive[dst]
    src = src[keep]
    dst = dst[keep]
    sums = torch.zeros_like(x)
    counts = torch.zeros(x.shape[0], dtype=x.dtype, device=x.device)
    if src.numel():
        sums.index_add_(0, src, x[dst])
        counts.index_add_(0, src, torch.ones_like(src, dtype=x.dtype))
    has_target = counts > 0
    targets = sums / counts.clamp_min(1).unsqueeze(-1)
    return targets, has_target
