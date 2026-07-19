import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GATEdgeClassifier(nn.Module):
    """GAT: dung co che attention -- KHAC GCN o cho tu hoc trong so quan trong khac nhau cho
    tung hang xom, va (quan trong) quyet dinh trong so do DUA VAO ca dac trung cua chinh canh
    noi toi hang xom (edge_dim), khong chi dua vao cau truc do thi don thuan. Neu khong co
    edge_dim, attention chi "nhin" duoc bac cua node, khong biet noi dung luong mang that.
    """

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        heads: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.convs = nn.ModuleList(
            [GATConv(node_in_dim, hidden_dim, heads=heads, dropout=dropout, edge_dim=edge_in_dim)]
        )
        for _ in range(num_layers - 1):
            self.convs.append(
                GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout, edge_dim=edge_in_dim)
            )
        self.dropout = dropout
        node_out_dim = hidden_dim * heads

        self.classifier = nn.Sequential(
            nn.Linear(node_out_dim * 2 + edge_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = F.elu(conv(x, edge_index, edge_attr))
            x = F.dropout(x, p=self.dropout, training=self.training)

        src, dst = edge_index
        edge_repr = torch.cat([x[src], x[dst], edge_attr], dim=1)
        return self.classifier(edge_repr)
