import torch
from torchdrug import core, models, transforms, data, layers
from torchdrug.layers import geometry
import os
from torchdrug.utils import comm
import sys
import logging
from itertools import islice
from torch.utils import data as torch_data
from utils import GODataset, TargetPrediction, pearsonr, spearmanr, Engine_WandB, sweep_run, MultipleBinaryClassification
os.environ['WANDB_EXECUTABLE'] = 'path/to/python'
import wandb

if __name__ == "__main__":
    wandb_project = 'GearNet-ProtGNN'
    wandb_name = 'Semi-Optimized No ESM' 
    truncate = 600
    save_name = f"geanet-protgnn-noesm"
    num_epoch = 100
    hyperparameter = False


    # GENERATE and LOAD Dataset
    
    protein_view_transform = transforms.ProteinView(view='residue')

    if truncate:
        truncuate_transform = transforms.TruncateProtein(max_length=truncate, random=False)
        transform = transforms.Compose([truncuate_transform, protein_view_transform])
    else:
        transform = protein_view_transform

    embed_file = '..Data/embeddings/protgnn_finetuned_noesm.pkl'
    map_file = 'helper_files/mapping_file.csv'
    dataset = GODataset("protein-datasets/", embed_file, map_file, transform=transform, atom_feature=None, bond_feature=None)
    train_set, valid_set, test_set = dataset.split()

    module = sys.modules[__name__]
    logger = logging.getLogger(__name__)


    if hyperparameter: 
        sweep_config = {
            'method': 'bayes',  # or 'grid', 'random'
            'metric': {
                'name': 'Valid Epoch/huber loss',  
                'goal': 'minimize'   
            },
            'parameters': {
                'epochs': {
                    'values': [20, 50, 100]  
                },
                'learning_rate': {
                    'min': 1e-5,
                    'max': 1e-3
                },
                'num_layers': {
                    'values': [2, 3, 4, 5]  
                },
                'batch_size': {
                    'values': [4, 16, 32, 64]  
                }
            }
        }

        sweep_id = wandb.sweep(sweep_config, project='GearNet-ProtGNN_sweep')
        wandb.agent(sweep_id, function=sweep_run, count=10, project='GearNet-ProtGNN_sweep')


    else:
        # Set models
        graph_construction_model = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()], 
                                                        edge_layers=[geometry.SpatialEdge(radius=10.0, min_distance=5),
                                                                    geometry.KNNEdge(k=10, min_distance=5),
                                                                    geometry.SequentialEdge(max_distance=2)],
                                                        edge_feature="gearnet")
        gearnet_edge = models.GearNet(input_dim=21, hidden_dims=[512, 512, 512, 512, 512, 512], 
                                    num_relation=7, edge_input_dim=59, num_angle_bin=8,
                                    batch_norm=True, concat_hidden=True, short_cut=True, readout="sum")

        # Set Task
        embedding = len(dataset[0]['TxGNN Embeddings'])
        task = TargetPrediction(gearnet_edge, embedding, graph_construction_model=graph_construction_model, num_mlp_layer=5,
                                            task=[_ for _ in range(len(dataset.tasks))], criterion="huber", metric=["auprc@micro", "r2"])
        

        # Train model
        wandb.finish()
        optimizer = torch.optim.Adam(task.parameters(), lr=1e-3)
        
        solver = Engine_WandB(task, train_set, valid_set, test_set, optimizer,
                            gpus=[0], batch_size=32, logger='wandb', log_interval=100, 
                            wandb_name=wandb_name, wandb_project=wandb_project)
        solver.load("models/mc_gearnet_edge.pth", load_optimizer=False, strict=False)
        solver.train(num_epoch=num_epoch, save_name = "models/{save_name}", save_epochs=200)
        solver.save(f"models/{save_name}.pth")
        solver.evaluate('test')