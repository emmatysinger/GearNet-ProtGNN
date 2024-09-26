import torch
from torchdrug import core, models, tasks, datasets, utils, transforms, data, layers, metrics
from torchdrug.layers import geometry
from torch.nn import functional as F
import pickle
import pandas as pd
import os
from torchdrug.utils import comm, pretty
import sys
import logging
from itertools import islice
from torch.utils import data as torch_data
os.environ['WANDB_EXECUTABLE'] = 'path/to/python'
import wandb
import numpy as np
from utils import Engine_WandB, MultipleBinaryClassification

if __name__ == "__main__":
    # PATHS TO MODELS #
    model_path = "/path/to/models/model_name.pth"

    num_epochs = 25
    truncate = 600
    dataset_type = 'GO'
    branch = 'MF'
    freeze = True

    if dataset_type == 'GO':
        save_extension = f"GO-{branch}"
    else:
        save_extension = dataset_type
    save_name = f"GearNet-{save_extension}_freeze"
    wandb_name = f"GearNet-{save_extension} (freeze)"
    wandb_project = "Downstream Tasks"
    load_path = model_path
    model_only = False # True for gearnet
    gearnet_protgnn = True
    remove_mlp = not gearnet_protgnn #keep mlp layers for gearnet_protgnn model
    

    protein_view_transform = transforms.ProteinView(view='residue')
    if truncate:
        truncuate_transform = transforms.TruncateProtein(max_length=truncate, random=False)
        transform = transforms.Compose([truncuate_transform, protein_view_transform])
    else:
        transform = protein_view_transform 

    if dataset_type == 'EC':
        dataset = datasets.EnzymeCommission("protein-datasets/", transform=protein_view_transform, atom_feature=None, bond_feature=None)
    elif dataset_type == 'GO':
        dataset = datasets.GeneOntology("protein-datasets/", branch=branch, transform=transform, atom_feature=None, bond_feature=None)
    train_set, valid_set, test_set = dataset.split()

    graph_construction_model = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()], 
                                                    edge_layers=[geometry.SpatialEdge(radius=10.0, min_distance=5),
                                                                    geometry.KNNEdge(k=10, min_distance=5),
                                                                    geometry.SequentialEdge(max_distance=2)],
                                                    edge_feature="gearnet")
    gearnet_edge = models.GearNet(input_dim=21, hidden_dims=[512, 512, 512, 512, 512, 512], 
                                num_relation=7, edge_input_dim=59, num_angle_bin=8,
                                batch_norm=True, concat_hidden=True, short_cut=True, readout="sum")
    
    

    task = MultipleBinaryClassification(gearnet_edge, graph_construction_model=graph_construction_model, num_mlp_layer=3, batch_norm=True, dropout=0.3,
                                                task=[_ for _ in range(len(dataset.tasks))], criterion="bce", metric=["auprc@micro", "f1_max"], gearnet_protgnn=gearnet_protgnn, embedding_dim=1024)

    optimizer = torch.optim.Adam(task.parameters(), lr=1e-4, weight_decay=1e-5)
    solver = Engine_WandB(task, train_set, valid_set, test_set, optimizer,
                        gpus=[0], batch_size=24, logger='wandb', log_interval=100, 
                        wandb_name=wandb_name, wandb_project=wandb_project)
    if load_path:
        solver.load(load_path, load_optimizer=False, strict=False, model_only=model_only, remove_mlp=remove_mlp)

    if freeze:
        for param in solver.model.model.parameters():
            param.requires_grad = False

    solver.train(num_epoch=num_epochs, save_name = f"models/{save_name}", save_epochs=num_epochs)
    solver.evaluate("test")