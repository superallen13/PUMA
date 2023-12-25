# For graph learning
import torch
from torch_geometric import seed_everything
from torch_sparse import SparseTensor
from torch_geometric.transforms import RandomNodeSplit

# Utility
import os
import argparse
import sys
import numpy as np
from utilities import *
from data_stream import Streaming
from torch_geometric.data import Batch
from backbones.gnn import train_node_classifier, train_node_classifier_with_validation, train_node_classifier_batch, eval_node_classifier

def _add_pseudo_labels(memory_bank, stream, model, lr, epoch, retrain, device, method, IL, threshold, batch):
    tasks = stream.tasks
    model.initialize()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    for k in range(len(tasks)):
        task = tasks[k]
        replayed_graphs = Batch.from_data_list(memory_bank[:k+1])
        replayed_graphs.to(device, "x", "y", "adj_t")
        max_cls = torch.unique(replayed_graphs.y)[-1]

        if retrain:
            model.initialize()
            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)

        if method == "cgm" or method == "ergnn":
            if batch:
                batches = memory_bank[:k+1]
                for data in batches:
                    data.to(device, "x", "y", "adj_t")
                model = train_node_classifier_batch(model, batches, opt, weight=None, n_epoch=epoch, incremental_cls=(0, max_cls+1), with_edge=False)
            else:
                model = train_node_classifier(model, replayed_graphs, opt, weight=None, n_epoch=epoch, incremental_cls=(0, max_cls+1), with_edge=False)
        else:
            if batch:
                batches = memory_bank[:k+1]
                for data in batches:
                    data.to(device, "x", "y", "adj_t")
                model = train_node_classifier_batch(model, batches, opt, weight=None, n_epoch=epoch, incremental_cls=(0, max_cls+1), with_edge=True)
            else:
                model = train_node_classifier(model, replayed_graphs, opt, weight=None, n_epoch=epoch, incremental_cls=(0, max_cls+1), with_edge=True)

        if IL == "taskIL":
            logits = model(task.to(device), with_edge=True)[:, 2*k:2*k+2]
        elif IL == "classIL":
            logits = model(task.to(device), with_edge=True)[:, :2*k+2]
            
        m = torch.nn.Softmax(dim=1)
        prob, pseudo_labels = m(logits).max(dim=1)
        if IL == "taskIL":
            pseudo_labels = pseudo_labels + 2 * k
        confident_select = (prob >= threshold).logical_and(task.test_mask)

        task.pseudo_labels = pseudo_labels  # Modify the labeled data.
        task.confident_select = confident_select

def main():
    parser = argparse.ArgumentParser()
    # Arguments for data.
    parser.add_argument('--dataset-name', type=str, default="corafull")
    parser.add_argument('--cls-per-task', type=int, default=2)
    parser.add_argument('--data-dir', type=str, default="./data")
    parser.add_argument('--result-path', type=str, default="./results")

    # Argumnets for CGL methods.
    parser.add_argument('--tim', action='store_true')
    parser.add_argument('--cgl-method', type=str, default="cgm")
    parser.add_argument('--cls-epoch', type=int, default=200)
    parser.add_argument('--hidden', type=int, default=512)
    parser.add_argument('--layers-num', type=int, default=2)
    parser.add_argument('--budget', type=int, default=2)
    parser.add_argument('--m-update', type=str, default="all")
    parser.add_argument('--pseudo-label', action='store_true')
    parser.add_argument('--edge-free', action='store_true')
    parser.add_argument('--cgm-args', type=str, default="{}")
    parser.add_argument('--ewc-args', type=str, default="{'memory_strength': 100000.}")
    parser.add_argument('--mas-args', type=str, default="{'memory_strength': 10000.}")
    parser.add_argument('--gem-args', type=str, default="{'memory_strength': 0.5, 'n_memories': 20}")
    parser.add_argument('--twp-args', type=str, default="{'lambda_l': 10000., 'lambda_t': 10000., 'beta': 0.01}")
    parser.add_argument('--lwf-args', type=str, default="{'lambda_dist': 10., 'T': 20.}")
    parser.add_argument('--IL', type=str, default="classIL")
    parser.add_argument('--batch', action='store_true')
    parser.add_argument('--threshold', type=float, default=0.)

    # Others
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--retrain', action='store_true')
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--rewrite', action='store_true')
    parser.add_argument('--early-stop', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--wandb', action='store_true')

    args = parser.parse_args()
    print(args)

    seed_everything(args.seed)
    
    # Make the streaming data.
    dataset = get_dataset(args)
    if not os.path.exists(os.path.join(args.data_dir, "streaming")):
        os.mkdir(os.path.join(args.data_dir, "streaming"))
    task_file = os.path.join(args.data_dir, "streaming", f"{args.dataset_name}.streaming")
    if os.path.exists(task_file):
        data_stream = torch.load(task_file)
    else:
        data_stream = Streaming(args.cls_per_task, dataset)
        torch.save(data_stream, task_file)

    # Handle the pseudo labels.
    if args.pseudo_label and not args.evaluate and (not os.path.exists(memory_bank_file) or args.rewrite):
        model = get_backbone_model(dataset, data_stream, args)
        cgl_model = get_cgl_model(model, data_stream, args)
        cgl_model.pseudo_label = False
        cgl_model.observer()
        _add_pseudo_labels(cgl_model.memory_bank, data_stream, cgl_model.model, args.lr, args.cls_epoch, args.retrain, args.device, args.cgl_method, args.IL, args.threshold, args.batch)

    APs = []
    AFs = []
    mAPs = []
    Ps = []
    memory_bank_name = get_memory_bank_name(args)
    # Strat the training.
    for i in range(args.repeat):
        torch.cuda.empty_cache()
        if args.cgl_method in ["bare", "ewc", "mas", "gem", "twp", "lwf"]:
            model = get_backbone_model(dataset, data_stream, args)
            cgl_model = get_cgl_model(model, data_stream, args)
            AP, mAP, AF = cgl_model.observer(args.cls_epoch, args.IL)
            APs.append(AP)
            AFs.append(AF)
            mAPs.append(mAP)
        else:
            model = get_backbone_model(dataset, data_stream, args)
            cgl_model = get_cgl_model(model, data_stream, args)
            memory_bank_file = os.path.join(args.result_path, "memory_bank", f"{memory_bank_name}_{i}.pt")
            if os.path.exists(memory_bank_file) and not args.rewrite:
                cgl_model.memory_bank = torch.load(memory_bank_file)

            AP, mAP, AF, performace_matrix = cgl_model.train_and_evaluate(args.tim, args.retrain, args.cls_epoch, args.cgl_method, args.IL, args.edge_free, args.batch, args.evaluate)
            APs.append(AP)
            AFs.append(AF)
            mAPs.append(mAP)
            Ps.append(performace_matrix)

            for memory in cgl_model.memory_bank:
                memory.to("cpu")
            
            if not os.path.exists(os.path.join(args.result_path, "memory_bank")):
                os.mkdir(os.path.join(args.result_path, "memory_bank"))
                
            if not os.path.exists(memory_bank_file) or args.rewrite:
                torch.save(cgl_model.memory_bank, memory_bank_file)
        
        for task in cgl_model.tasks:
            task.to("cpu")
        
    if args.evaluate:
        print(f"{np.mean(APs):.1f}±{np.std(APs, ddof=1):.1f}", flush=True)
        print(f"{np.mean(mAPs):.1f}±{np.std(mAPs, ddof=1):.1f}", flush=True)
        print(f"{np.mean(AFs):.1f}±{np.std(AFs, ddof=1):.1f}", flush=True)


if __name__ == '__main__':
    main()