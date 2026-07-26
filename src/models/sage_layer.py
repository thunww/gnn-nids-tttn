import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class EGraphSAGEConv(MessagePassing):
    """E-GraphSAGE (Lo et al., 2021, arXiv:2103.16329) -- SAGEConv cua PyG khong ho tro
    edge_dim nen phai tu viet: message tu hang xom v ket hop CA dac trung canh noi v->u
    (khac GraphSAGE goc chi dung dac trung node), sum lai roi cap nhat embedding u.
    Cong thuc dung dung docs/graphsage_plan.md muc 3:
        message: phi(x_v, e_vu) = W1 . [x_v ; e_vu]
        aggregate: a = sum_{v in N(u)} phi_v
        update: h_u = sigma(W2 . [x_u ; a])
    """

    def __init__(self, node_in_dim: int, edge_in_dim: int, out_dim: int):
        super().__init__(aggr="add")
        self.lin_message = nn.Linear(node_in_dim + edge_in_dim, out_dim)
        self.lin_update = nn.Linear(node_in_dim + out_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.lin_message(torch.cat([x_j, edge_attr], dim=1))

    def update(self, aggr_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.lin_update(torch.cat([x, aggr_out], dim=1)))
