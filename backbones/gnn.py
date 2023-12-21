import torch
import torch.nn.functional as F
import copy


class GNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([])

    def initialize(self):
        for layer in self.layers:
            layer.reset_parameters()
    
    def forward(self, data):
        x, adj_t = data.x, data.adj_t
        for layer in self.layers[:-1]:
            x = layer(x, adj_t)
            x = F.relu(x)
        x = self.layers[-1](x, adj_t)
        return x

    def encode(self, x, adj_t):
        self.eval()
        for i, layer in enumerate(self.layers[:-1]):  # without the FC layer.
            x = layer(x, adj_t)
            x = F.relu(x)
        return x
    
    def encode_noise(self, x, adj_t):
        self.eval()
        for i, layer in enumerate(self.layers):
            x = layer(x, adj_t)
            if i != len(self.layers) - 1:
                x = F.relu(x)
        random_noise = torch.rand_like(x).cuda()
        x += torch.sign(x) * F.normalize(random_noise, dim=-1) * 0.1
        return x

def train_node_classifier(model, data, optimizer, weight=None, n_epoch=200, incremental_cls=None, with_edge=True):
    model.train()
    ce = torch.nn.CrossEntropyLoss(weight=weight)
    for _ in range(n_epoch):
        if incremental_cls:
            out = model(data, with_edge=with_edge)[:, incremental_cls[0]:incremental_cls[1]]
        else:
            out = model(data, with_edge=with_edge)
        loss = ce(out[data.train_mask], data.y[data.train_mask])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

def train_node_classifier_batch(model, batches, optimizer, weight=None, n_epoch=200, incremental_cls=None, with_edge=True):
    model.train()
    ce = torch.nn.CrossEntropyLoss(weight=weight)
    for _ in range(n_epoch):
        loss = torch.tensor(0.).to(batches[0].x.device)
        for data in batches:
            if incremental_cls:
                out = model(data, with_edge=with_edge)[:, incremental_cls[0]:incremental_cls[1]]
            else:
                out = model(data, with_edge=with_edge)
            loss += ce(out[data.train_mask], data.y[data.train_mask])
        loss /= len(batches)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model

def train_node_classifier_with_validation(model, data, validation_graph, n_epochs_stop, optimizer, weight=None, n_epoch=200, incremental_cls=None, with_edge=True):
    import wandb
    model.train()
    ce = torch.nn.CrossEntropyLoss(weight=weight)
    max_acc_valid = 0.
    epochs_no_improve = 0
    classes = torch.unique(validation_graph.y)

    for _ in range(n_epoch):
        if incremental_cls:
            out = model(data, with_edge=with_edge)[:, 0:incremental_cls[1]]
        else:
            out = model(data, with_edge=with_edge)
        loss = ce(out[data.train_mask], data.y[data.train_mask])
        
        acc_valid = 0.
        count = 0

        pred = model(validation_graph)[validation_graph.val_mask]
        for cls in classes:
            count += 1
            pred_cls = pred[validation_graph.y[validation_graph.val_mask] == cls]
            acc_valid += int((pred_cls.argmax(dim=1) == cls).sum()) / int((validation_graph.y[validation_graph.val_mask] == cls).sum())
        acc_valid /= count

        # wandb.log({"loss": loss, "loss val": loss_valid, "loss test": loss_test, 
        #            "acc val": acc_valid, "acc test": acc_test})

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if acc_valid > max_acc_valid:
                max_acc_valid = acc_valid
                best_model = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            # Check early stopping condition
            if epochs_no_improve == n_epochs_stop:
                print(f'Early stopping at epoch {_ - n_epochs_stop}', flush=True)
                model.load_state_dict(best_model)
                return model
    return model


def eval_node_classifier(model, data, incremental_cls=None):
    model.eval()
    pred = model(data)[data.test_mask, incremental_cls[0]:incremental_cls[1]].argmax(dim=1)
    correct = (pred == data.y[data.test_mask]-incremental_cls[0]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    return acc
