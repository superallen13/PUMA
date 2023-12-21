# DL and GL
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_sparse import SparseTensor
import random

# Own modules
from methods.replay import Replay
from backbones.encoder import Encoder

# Utilities
from .utility import *
import random


class CGM(Replay):
    def __init__(self, model, lr, tasks, budget, m_update, pseudo_label, device, args):
        super().__init__(model, lr, tasks, budget, m_update, pseudo_label, device)
        self.update_epoch = args['update_epoch']
        self.n_encoders = args['n_encoders']
        self.feat_lr = args['feat_lr']
        self.hid_dim = args['hid_dim']
        self.emb_dim = args['emb_dim']
        self.n_layers = args['n_layers']
        self.feat_init = "randomChoice"
        self.feat_init = args["feat_init"]
        self.hop = args['hop']
        self.activation = args['activation']
        self.otp = args['otp']
        
    def memorize(self, task, budgets):
        labels_cond = []
        for i, cls in enumerate(task.classes):
            labels_cond += [cls] * budgets[i]
        labels_cond = torch.tensor(labels_cond)

        feat_cond = torch.nn.Parameter(torch.FloatTensor(sum(budgets), task.num_features))
        feat_cond = self._initialize_feature(task, budgets, feat_cond, self.feat_init)

        replayed_graph = self._condense(task, feat_cond, labels_cond, budgets)
        return replayed_graph
    
    def _initialize_feature(self, task, budgets, feat_cond, method="randomChoice"):
        if method == "randomNoise":
            torch.nn.init.xavier_uniform_(feat_cond)
        elif method == "randomChoice":
            sampled_ids = []
            for i, cls in enumerate(task.classes):
                train_mask = task.train_mask
                train_mask_at_cls = (task.y == cls).logical_and(train_mask)
                ids_at_cls = train_mask_at_cls.nonzero(as_tuple=True)[0].tolist()
                sampled_ids += random.choices(ids_at_cls, k=budgets[i])
            sampled_feat = task.x[sampled_ids]
            feat_cond.data.copy_(sampled_feat)
        return feat_cond

    def _condense(self, task, feat_cond, labels_cond, budgets):
        cls_train_masks = []
        for cls in task.classes:
            if self.pseudo_label:
                cls_train_mask = (task.y == cls).logical_and(task.train_mask)
                cls_pseudo_mask = (task.pseudo_labels == cls).logical_and(task.confident_select)
                cls_train_masks.append(cls_train_mask.logical_or(cls_pseudo_mask))
            else:
                cls_train_masks.append((task.y == cls).logical_and(task.train_mask))  

        max_cls = torch.max(task.y)
        encoder = Encoder(task.num_features, self.hid_dim, max_cls+1, self.n_layers, self.hop, self.activation).to(self.device)

        for encoder_id in range(self.n_encoders):
            encoder.initialize()
            opt_feat = torch.optim.Adam([feat_cond], lr=self.feat_lr)

            with torch.no_grad():
                emb_real = encoder.encode(task.x.to(self.device), task.adj_t.to(self.device), cache=self.otp)
                emb_real = F.normalize(emb_real)
                
            for inner in range(self.update_epoch):
                emb_cond = encoder.encode_without_e(feat_cond.to(self.device), subnet=None)
                emb_cond = F.normalize(emb_cond)

                loss = torch.tensor(0.).to(self.device)
                for i, cls in enumerate(task.classes):
                    real_emb_at_class = emb_real[cls_train_masks[i]]
                    cond_emb_at_class = emb_cond[labels_cond == cls]
                    dist = torch.mean(real_emb_at_class, 0) - torch.mean(cond_emb_at_class, 0)
                    loss += torch.sum(dist ** 2)
                opt_feat.zero_grad()
                loss.backward()
                opt_feat.step()
                
        # Wrap the graph data object
        self_loops = SparseTensor.eye(sum(budgets), sum(budgets)).t()
        replayed_graph = Data(x=feat_cond.detach().cpu(), 
                              y=labels_cond.detach().cpu(), 
                              adj_t=self_loops)
        replayed_graph.train_mask = torch.ones(sum(budgets), dtype=torch.bool)
        return replayed_graph