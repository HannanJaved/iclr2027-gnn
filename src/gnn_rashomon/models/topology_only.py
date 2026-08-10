from __future__ import annotations

try:
    import torch
    from torch import nn
    from torch_geometric.nn import GCNConv
except ModuleNotFoundError:  # pragma: no cover - exercised only without optional deps
    torch = None
    nn = None
    GCNConv = None


if nn is None:

    class TopologyOnlyGCN:  # type: ignore[no-redef]
        def __init__(
            self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float
        ) -> None:
            raise RuntimeError("TopologyOnlyGCN requires torch and torch-geometric.")

else:

    class TopologyOnlyGCN(nn.Module):  # type: ignore[no-redef]
        def __init__(
            self, in_channels: int, hidden_channels: int, out_channels: int, dropout: float
        ) -> None:
            if torch is None or GCNConv is None:
                raise RuntimeError("TopologyOnlyGCN requires torch and torch-geometric.")
            super().__init__()
            del in_channels
            self.conv1 = GCNConv(1, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)
            self.dropout = float(dropout)

        def forward(self, x: "torch.Tensor", edge_index: "torch.Tensor") -> "torch.Tensor":
            x = x.new_ones((x.size(0), 1))
            x = self.conv1(x, edge_index).relu()
            x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
            return self.conv2(x, edge_index)
