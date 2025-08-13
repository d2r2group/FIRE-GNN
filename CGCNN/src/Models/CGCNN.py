import torch
import torch.nn as nn
from typing import Literal
from torch_geometric.nn.conv import CGConv
from torch_scatter import scatter

class CGCNN(nn.Module):
    def __init__(self, 
                 output_dim, 
                 h_dim=128, 
                 num_graph_layers=5, 
                 num_fc_layers=3, 
                 reduce: Literal["mean", "max"] = "mean"
                ):
        super().__init__()
        self.atom_embedding = nn.Embedding(119, h_dim)

        self.graph_layers = nn.ModuleList([
            CGConv(h_dim, dim=1) for _ in range(num_graph_layers)
        ])

        self.output_layers = nn.ModuleList([
            nn.Linear(h_dim, h_dim) for _ in range(num_fc_layers)
        ])
        self.output_layers.append(nn.Linear(h_dim, output_dim))

        self.reduce = reduce

    def forward(self, data):
        x = self.atom_embedding(data.atoms)
        edge_attr = torch.norm(data.pos[data.edge_index[0]]-data.pos[data.edge_index[1]], keepdim=True, dim=-1)

        for layer in self.graph_layers:
            x = layer(x, data.edge_index, edge_attr)

        x = scatter(x, data.batch, dim=0, reduce=self.reduce)

        for layer in self.output_layers:
            x = layer(x)
        
        return x
