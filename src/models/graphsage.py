import torch
import torch.nn as nn
import torch.nn.functional as F

from models.sage_layer import EGraphSAGEConv


class GraphSAGEEdgeClassifier(nn.Module):
    """E-GraphSAGE: moi lop tinh "thong diep" tu hang xom bang cach ket hop dac trung hang
    xom + dac trung canh noi toi no, tong hop (sum) roi cap nhat embedding node hien tai --
    khac GCN (khong dung dac trung canh trong message passing) va GAT (dung attention thay
    vi tong hop don gian). Ghep embedding 2 dau canh + dac trung canh -> phan loai da lop,
    giong het pattern GCN/GAT (xem gcn.py, gat.py).
    """

    def __init__(
        self,
        node_in_dim: int,
        edge_in_dim: int,
        hidden_dim: int,
        num_classes: int,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.convs = nn.ModuleList([EGraphSAGEConv(node_in_dim, edge_in_dim, hidden_dim)])
        for _ in range(num_layers - 1):
            self.convs.append(EGraphSAGEConv(hidden_dim, edge_in_dim, hidden_dim))
        self.dropout = dropout

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.dropout(x, p=self.dropout, training=self.training)

        src, dst = edge_index
        edge_repr = torch.cat([x[src], x[dst], edge_attr], dim=1)
        return self.classifier(edge_repr)
