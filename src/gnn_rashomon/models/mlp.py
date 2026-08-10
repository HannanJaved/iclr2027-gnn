from __future__ import annotations

try:
    import torch
    from torch import nn
except ModuleNotFoundError:  # pragma: no cover - exercised only without optional deps
    torch = None
    nn = None


if nn is None:

    class MLP:  # type: ignore[no-redef]
        def __init__(
            self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float
        ) -> None:
            raise RuntimeError("MLP requires torch.")

else:

    class MLP(nn.Module):  # type: ignore[no-redef]
        def __init__(
            self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float
        ) -> None:
            super().__init__()
            self.lin1 = nn.Linear(in_channels, hidden_channels)
            self.lin2 = nn.Linear(hidden_channels, out_channels)
            self.dropout = float(dropout)

        def forward(self, x: "torch.Tensor", edge_index: "torch.Tensor") -> "torch.Tensor":
            del edge_index
            x = self.lin1(x).relu()
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
            return self.lin2(x)
