from methods.replay import Replay
import torch
from torch_geometric.data import Data
from torch_sparse import SparseTensor

class Joint(Replay):
    def __init__(self, model, lr, tasks, budget, m_update, pseudo_label, device):
        super().__init__(model, lr, tasks, budget, m_update, pseudo_label, device)

    def memorize(self, task, budgets):
        if self.pseudo_label:
            self_loops = SparseTensor.eye(task.x.shape[0], task.x.shape[0]).t()
            replayed_graph = Data(x=task.x.detach().cpu(), y=task.y, adj_t=self_loops)
            replayed_graph.train_mask = torch.ones(task.x.shape[0], dtype=torch.bool)
            replayed_graph.y = task.pseudo_labels
            return replayed_graph
        else:
            return task