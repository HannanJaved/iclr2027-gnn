from __future__ import annotations

try:
    import torch
    from torch import nn
    from torch_geometric.nn import GATConv
except ModuleNotFoundError:  # pragma: no cover - exercised only without optional deps
    torch = None
    nn = None
    GATConv = None


if nn is None:

    class GAT:  # type: ignore[no-redef]
        def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            dropout: float,
            heads: int = 4,
        ) -> None:
            raise RuntimeError("GAT requires torch and torch-geometric.")

else:

    class GAT(nn.Module):  # type: ignore[no-redef]
        def __init__(
            self,
            in_channels: int,
            hidden_channels: int,
            out_channels: int,
            dropout: float,
            heads: int = 4,
        ) -> None:
            if torch is None or GATConv is None:
                raise RuntimeError("GAT requires torch and torch-geometric.")
            super().__init__()
            self.conv1 = GATConv(
                in_channels,
                hidden_channels,
                heads=heads,
                dropout=dropout,
            )
            self.conv2 = GATConv(
                hidden_channels * heads,
                out_channels,
                heads=1,
                concat=False,
                dropout=dropout,
            )
            self.dropout = float(dropout)

        def forward(self, x: "torch.Tensor", edge_index: "torch.Tensor") -> "torch.Tensor":
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
            x = torch.nn.functional.elu(self.conv1(x, edge_index))
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
            return self.conv2(x, edge_index)
