import torch
from torch_geometric.data import Batch
from backbones.gnn import train_node_classifier, eval_node_classifier, train_node_classifier_batch
import copy
import sys

# Utilities
from progressbar import progressbar
from methods.utility import get_graph_class_ratio
from backbones.gcn import GCN
import time

class Replay():
    def __init__(self, model, lr, tasks, budget, m_update, pseudo_label, device):
        super().__init__()
        self.model = model
        self.lr = lr
        self.tasks = tasks
        self.budgets = self._assign_buget_per_cls(budget)
        self.device = device
        self.memory_bank = []
        self.strategy = m_update
        self.pseudo_label = pseudo_label
    
    def train_and_evaluate(self, tim, retrain, epoch, method, IL, edge_free, batch, evaluate):
        tasks = self.tasks

        self.model.initialize()
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=5e-4)

        # Initialize the performance matrix.
        mAP = 0
        performace_matrix = torch.zeros(len(tasks), len(tasks))
        
        memorise_time = 0
        train_time = 0
        for k in progressbar(range(len(tasks)), redirect_stdout=True):
            task = tasks[k]
            start_time = time.time()
            if len(self.memory_bank) == k:
                replayed_graph = self.memorize(task, self.budgets[k]).to("cpu")
                self.memory_bank.append(replayed_graph)  # update memory bank
            memorise_time += time.time() - start_time
            if evaluate:
                if tim:
                    replayed_graphs = self.memory_bank[:k+1]
                    for replayed_graph in replayed_graphs:
                        replayed_graph.to("cpu")
                    replayed_graphs = Batch.from_data_list(replayed_graphs)
                else:
                    for memory in self.memory_bank[:k]:
                        memory.to("cpu")
                    replayed_graphs = self.memory_bank[:k] + [tasks[k].to("cpu")]
                    replayed_graphs = Batch.from_data_list(replayed_graphs)
                    
                replayed_graphs.to(self.device, "x", "y", "adj_t")
                max_cls = torch.max(replayed_graphs.y)

                if retrain:
                    self.model.initialize()
                    opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=5e-4)
                
                start_time = time.time()
                if edge_free:
                    if batch:
                        batches = self.memory_bank[:k+1]
                        for data in batches:
                            data.to(self.device, "x", "y", "adj_t")
                        self.model = train_node_classifier_batch(self.model, batches, opt, weight=None, n_epoch=epoch, incremental_cls=(0, max_cls+1), with_edge=False)
                    else:
                        self.model = train_node_classifier(self.model, replayed_graphs, opt, weight=None, n_epoch=epoch, incremental_cls=(0, max_cls+1), with_edge=False)
                else:
                    if batch:
                        batches = self.memory_bank[:k+1]
                        for data in batches:
                            data.to(self.device, "x", "y", "adj_t")
                        self.model = train_node_classifier_batch(self.model, batches, opt, weight=None, n_epoch=epoch, incremental_cls=(0, max_cls+1), with_edge=True)
                    else:
                        self.model = train_node_classifier(self.model, replayed_graphs, opt, weight=None, n_epoch=epoch, incremental_cls=(0, max_cls+1), with_edge=True)
                train_time += time.time() - start_time

                # Test the model from task 0 to task k
                self.model.eval()
                accs = []
                AF = 0
                for k_ in range(k + 1):
                    test_graph = tasks[k_].to(self.device, "x", "y", "adj_t")
                    pred_ = self.model(test_graph)[test_graph.test_mask, 0:max_cls+1]
                    if IL == "taskIL":
                        pred = pred_[:, k_*2:k_*2+2].argmax(dim=1) + k_ * 2
                    elif IL == "classIL":
                        pred = pred_.argmax(dim=1)
                    task_mask = (test_graph.y[test_graph.test_mask] == 2 * k_).logical_or(test_graph.y[test_graph.test_mask] == 2 * k_ + 1)
                    correct = (pred[task_mask] == test_graph.y[test_graph.test_mask][task_mask]).sum()
                    acc = int(correct) / int(task_mask.sum()) * 100
                    accs.append(acc)
                    performace_matrix[k, k_] = acc
                    print(f"T{k_} {acc:.2f}", end="|", flush=True)
                test_graph.to("cpu")
                AP = sum(accs) / len(accs)
                mAP += AP
                print(f"AP: {AP:.2f}", end=", ", flush=True)
                print(f"mAP: {mAP/(k+1):.2f}", end=", ", flush=True)
                for t in range(k):
                    AF += performace_matrix[k, t] - performace_matrix[t, t]
                AF = AF / k if k != 0 else AF
                print(f"AF: {AF:.2f}", flush=True)
        if evaluate:
            print(f"Memorisation time (s): {memorise_time}, Training time (s): {train_time}", flush=True)
            return AP, mAP/(k + 1), AF, performace_matrix
        else:
            print(f"Memorisation time (s): {memorise_time}", flush=True)
            return 0, 0, 0, performace_matrix

    def observer(self):
        tasks = self.tasks
        for k in progressbar(range(len(tasks)), redirect_stdout=True):
            task = tasks[k]
            replayed_graph = self.memorize(task, self.budgets[k])
            self.memory_bank.append(replayed_graph)  # update memory bank
    
    def memorize(self, task, budgets):
        raise NotImplementedError("Please implement this method!")
    
    def _assign_buget_per_cls(self, budget):
        budgets = []
        for task in self.tasks:
            if budget is None:
                budgets.append([])
            else:
                classes = torch.unique(task.y)
                budgets_at_task = []
                for cls in classes:
                    class_ratio = get_graph_class_ratio(task, cls)
                    replay_cls_size = int(budget * class_ratio)
                    if replay_cls_size == 0:
                        budgets_at_task.append(1)
                    else:
                        budgets_at_task.append(replay_cls_size)
                # Because using the int(), sometimes, there are still some nodes which are not be allocated a label.
                gap = budget - sum(budgets_at_task)

                for i in range(gap):
                    budgets_at_task[i % len(classes)] += 1
                budgets.append(budgets_at_task)
        return budgets

