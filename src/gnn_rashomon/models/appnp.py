from __future__ import annotations

try:
    import torch
    from torch import nn
    from torch_geometric.nn import APPNP as APPNPPropagation
except ModuleNotFoundError:  # pragma: no cover - exercised only without optional deps
    torch = None
    nn = None
    APPNPPropagation = None


if nn is None:

    class APPNP:  # type: ignore[no-redef]
        def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            dropout: float,
            k: int = 10,
            alpha: float = 0.1,
        ) -> None:
            raise RuntimeError("APPNP requires torch and torch-geometric.")

else:

    class APPNP(nn.Module):  # type: ignore[no-redef]
        def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            dropout: float,
            k: int = 10,
            alpha: float = 0.1,
        ) -> None:
            if torch is None or APPNPPropagation is None:
                raise RuntimeError("APPNP requires torch and torch-geometric.")
            super().__init__()
            self.lin1 = nn.Linear(in_channels, hidden_channels)
            self.lin2 = nn.Linear(hidden_channels, out_channels)
            self.propagation = APPNPPropagation(K=k, alpha=alpha, dropout=dropout)
            self.dropout = float(dropout)

        def forward(self, x: "torch.Tensor", edge_index: "torch.Tensor") -> "torch.Tensor":
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
            x = self.lin1(x).relu()
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
            x = self.lin2(x)
            return self.propagation(x, edge_index)
