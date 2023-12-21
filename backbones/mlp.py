import torch
from backbones.gnn import GNN
import torch.nn.functional as F


class MLP(GNN):
    def __init__(self, nin, nhid, nout, nlayers):
        super().__init__()
        if nlayers == 1:
            self.layers.append(torch.nn.Linear(nin, nout))
        else:
            self.layers.append(torch.nn.Linear(nin, nhid))  # input layers
            for _ in range(nlayers - 2):
                self.layers.append(torch.nn.Linear(nhid, nhid))  # hidden layers
            self.layers.append(torch.nn.Linear(nhid, nout))  # output layers

    def forward(self, data):
        x = data.x
        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.relu(x)
        x = self.layers[-1](x)
        return x