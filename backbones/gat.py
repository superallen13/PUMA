import torch
from backbones.gnn import GNN
from torch_sparse import SparseTensor
from torch_geometric.nn import GATConv
import torch.nn.functional as F


class GAT(GNN):
    def __init__(self, nin, nhid, nout, nlayers, concat=False, heads=8):
        super().__init__()
        if nlayers == 1:
            self.layers.append(GATConv(nin, nout, concat=concat, heads=1))
        else:
            self.layers.append(GATConv(nin, nhid, concat=concat, heads=heads))  # input layers
            for _ in range(nlayers - 2):
                self.layers.append(GATConv(nhid, nhid, concat=concat, heads=heads))  # hidden layers
            self.layers.append(GATConv(nhid, nout, concat=concat, heads=1))  # output layers
    
    def forward(self, data):
        x, adj_t = data.x, data.adj_t
        for layer in self.layers[:-1]:
            x = layer(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        x, att_w = self.layers[-1](x, adj_t, return_attention_weights=True)
        return x, att_w

def train_node_classifier(model, data, optimizer, weight=None, n_epoch=200, incremental_cls=None):
    model.train()
    ce = torch.nn.CrossEntropyLoss(weight=weight)
    for epoch in range(n_epoch):
        out, _ = model(data)
        if incremental_cls:
            out = out[:, 0:incremental_cls[1]]
        loss = ce(out[data.train_mask], data.y[data.train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

def eval_node_classifier(model, data, incremental_cls=None):
    model.eval()
    out, _ = model(data)
    if incremental_cls:
        pred = out[data.test_mask, incremental_cls[0]:incremental_cls[1]].argmax(dim=1)
        correct = (pred == data.y[data.test_mask]-incremental_cls[0]).sum()
    else:
        pred = out[data.test_mask].argmax(dim=1)
        correct = (pred == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    return acc
