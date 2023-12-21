import torch
from backbones.gnn import GNN
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import spmm


def gcn_norm(adj_t):
    from torch_sparse import SparseTensor, fill_diag, matmul, mul
    from torch_sparse import sum as sparsesum
    adj_t = fill_diag(adj_t, 1.)  # add self-loops.
    
    # Normalization.
    deg = sparsesum(adj_t, dim=1)
    deg_inv_sqrt = deg.pow_(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
    adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
    adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))

    return adj_t


class Linear(GCNConv):
    def __init__(
        self,
        in_channels: int,
        out_channels: int
    ):
        super().__init__(in_channels, out_channels, bias=False)
    
    def forward(self, x, subnet=None):
        if subnet:
            return F.linear(x, self.lin.weight[subnet, :], self.lin.bias)
        else:
            return self.lin(x)
    

class Encoder(GNN):
    def __init__(self, nin, nhid, nout, nlayers, hop, activation=True):
        super().__init__()
        self.feat_agg = None
        self.hop = hop
        self.activation = activation  # True or False
        self.relu = torch.nn.LeakyReLU(0.1)
        self.feat_agg = None
        if nlayers == 1:
            self.layers.append(Linear(nin, nout))
        else:
            self.layers.append(Linear(nin, nhid))  # input layers
            for _ in range(nlayers - 2):
                self.layers.append(Linear(nhid, nhid))  # hidden layers
            self.layers.append(GCNConv(nhid, nout, bias=False))  # output layers

    # def initialize(self):
    #     torch.nn.init.normal_(self.layers[0].lin.weight)
    
    def forward(self, x, adj_t):
        for i, layer in enumerate(self.layers[:-1]):  # without the FC layer.
            adj_t = gcn_norm(adj_t)
            x = self.layers[0].propagate(adj_t, x=x)
            x = layer(x)
            if self.activation:
                x = F.relu(x)
        x = self.layers[-1](x, adj_t)
        return x
    
    def forward_without_e(self, x):
        for layer in self.layers[:-1]:  # without the FC layer.
            x = layer(x)
            if self.activation:
                x = F.relu(x)
        x = self.layers[-1].lin(x)
        return x
    
    def encode_without_e(self, x, subnet=None):
        outputs = []
        for layer in self.layers[:-1]:  # without the FC layer.
            x = layer(x, subnet=subnet)
            if self.activation:
                x = F.relu(x)
            outputs.append(x)
        return torch.cat(outputs, 1)

    def encode(self, x, adj_t, hop=1, cache=True):
        # Save the aggreated features for the first layer.
        if cache:
            if self.feat_agg is None:
                adj_t = gcn_norm(adj_t)
                for _ in range(self.hop):
                    x = spmm(adj_t, x)
                self.feat_agg = x
        else:
            adj_t = gcn_norm(adj_t)
            for _ in range(self.hop):
                x = spmm(adj_t, x)
            self.feat_agg = x
            # adj_t = torch.sigmoid(adj_t)
            # adj_t = (adj_t.t() + adj_t) / 2  # undirected graph
            # deg = torch.sum(adj_t, dim=1)
            # deg_inv_sqrt = deg.pow_(-0.5)
            # deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0.)
            # adj_t = torch.mul(adj_t, deg_inv_sqrt.view(-1, 1))
            # adj_t = torch.mul(adj_t, deg_inv_sqrt.view(1, -1))
            # self.feat_agg = torch.div(torch.mm(adj_t, x), deg.view(-1, 1))

        outputs = []
        for i, layer in enumerate(self.layers[:-1]):  # without the FC layer.
            if i == 0:
                x = self.feat_agg
            else:
                adj_t = gcn_norm(adj_t)
                x = spmm(adj_t, x)
            x = layer(x)
            if self.activation:
                x = F.relu(x)
            outputs.append(x)
        return torch.cat(outputs, 1)