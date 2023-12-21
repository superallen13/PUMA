from methods.replay import Replay
import torch
import random
from backbones.gcn import GCN
# from progressbar import progressbar
from torch_geometric.data import Data
from torch_sparse import SparseTensor
import copy

class RandomSubgraphSampler(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, center_node_budget, ids_per_cls):
        center_nodes_selected = self.node_sampler(ids_per_cls, center_node_budget)
        return center_nodes_selected

    def node_sampler(self, ids_per_cls_train, budget, max_ratio_per_cls = 1.0):
        store_ids = []
        budget_ = min(budget, int(max_ratio_per_cls * len(ids_per_cls_train)))
        store_ids.extend(random.sample(ids_per_cls_train, budget_))
        return store_ids

class DegreeBasedSampler(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, graph, center_node_budget, nei_budget, ids_per_cls):
        center_nodes_selected = self.node_sampler(ids_per_cls, center_node_budget)
        all_nodes_selected = self.nei_sampler(graph, center_nodes_selected, nei_budget)
        return center_nodes_selected, all_nodes_selected

    def node_sampler(self, ids_per_cls_train, budget, max_ratio_per_cls = 1.0):
        store_ids = []
        budget_ = min(budget, int(max_ratio_per_cls * len(ids_per_cls_train)))
        store_ids.extend(random.sample(ids_per_cls_train, budget_))
        return store_ids

    def nei_sampler(self, graph, center_nodes_selected, nei_budget):
        degrees = graph.in_degrees().float()
        total_degree = degrees.sum()
        probs = degrees / total_degree
        nodes_selected_current_hop = copy.deepcopy(center_nodes_selected)
        retained_nodes = copy.deepcopy(center_nodes_selected)

        for b in nei_budget:
            if b == 0:
                continue
            neighbors = list(set(graph.in_edges(nodes_selected_current_hop)[0].tolist()))
            neighbors = [n for n in neighbors if n not in retained_nodes]
            if len(neighbors) == 0:
                continue
            prob = probs[neighbors]
            sampled_neibs_ = torch.multinomial(prob, min(b, len(neighbors)), replacement=False).tolist()
            sampled_neibs = torch.tensor(neighbors)[sampled_neibs_]
            retained_nodes.extend(sampled_neibs.tolist())
        return list(set(retained_nodes))

class SSM(Replay):
    def __init__(self, model, lr, tasks, budget, m_update, pseudo_label, device):
        super().__init__(model, lr, tasks, budget, m_update, pseudo_label, device)

    def memorize(self, task, budgets):
        classes = torch.unique(task.y)
        ids_per_cls_train = []
        for cls in classes:
            if self.pseudo_label:
                cls_train_mask = (task.y == cls).logical_and(task.train_mask)
                cls_pseudo_mask = (task.pseudo_labels == cls).logical_and(task.confident_select)
                cls_train_mask = cls_train_mask.logical_or(cls_pseudo_mask)
            else:
                cls_train_mask = (task.y == cls).logical_and(task.train_mask)
            ids_per_cls_train.append(cls_train_mask.nonzero(as_tuple=True)[0].tolist())

        sampler = RandomSubgraphSampler()
        nodes_sampled = []
        for i, ids in enumerate(ids_per_cls_train):
            c_nodes_sampled = sampler(budgets[i], ids)
            nodes_sampled += c_nodes_sampled
        task.to("cpu")
        replayed_graph = task.subgraph(torch.tensor(nodes_sampled))
        edge_index = replayed_graph.edge_index
        adj = SparseTensor(row=edge_index[0], 
                           col=edge_index[1], 
                           sparse_sizes=(replayed_graph.num_nodes, replayed_graph.num_nodes))
        replayed_graph.adj_t = adj.t()
        return replayed_graph
        
