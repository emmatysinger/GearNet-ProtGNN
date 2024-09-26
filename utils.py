import torch
from torchdrug import core, models, tasks, datasets, utils, transforms, data, layers, metrics
from torchdrug.layers import geometry, functional
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

class GODataset(datasets.GeneOntology):
    url = "https://zenodo.org/record/6622158/files/GeneOntology.zip"
    md5 = "376be1f088cd1fe720e1eaafb701b5cb"
    branches = ["MF", "BP", "CC"]
    processed_file = "gene_ontology.pkl.gz"
    test_cutoffs = [0.3, 0.4, 0.5, 0.7, 0.95]
    def __init__(self, path, embed_file, map_file, branch="MF", test_cutoff=0.95, verbose=1, **kwargs):
        super().__init__(path, branch, test_cutoff, verbose, **kwargs)
        pdb_ids = [os.path.basename(pdb_file).split("_")[0][:4] for pdb_file in self.pdb_files]
        self.load_embeddings(embed_file, map_file, pdb_ids)


    def load_embeddings(self, embed_file, map_file, pdb_ids):
        with open(embed_file, 'rb') as f:
            embeddings = pickle.load(f)
        
        prot_embs = embeddings['gene/protein']
        mapping_df = pd.read_csv(map_file)

        mapped_emb_dict = {}
        for i, embed in enumerate(prot_embs):
            try:
                filtered_df = mapping_df.loc[mapping_df['idx'] == i, 'PDB ID']
                for pdb_value in filtered_df:
                    mapped_emb_dict[pdb_value] = embed
            except:
                continue

        filtered_pdb_ids = [pdb_id for pdb_id in pdb_ids if pdb_id in mapped_emb_dict]
        self.filter_pdb(filtered_pdb_ids)
        pdb_ids = [os.path.basename(pdb_file).split("_")[0][:4] for pdb_file in self.pdb_files]
        self.embeddings = [mapped_emb_dict[pdb_id] for pdb_id in pdb_ids]
        
        self.targets = {'TxGNN Embeddings':0}


    def filter_pdb(self, pdb_ids):
        pdb_ids = set(pdb_ids)
        sequences = []
        pdb_files = []
        data = []
        for sequence, pdb_file, protein in zip(self.sequences, self.pdb_files, self.data):
            if os.path.basename(pdb_file).split("_")[0][:4] in pdb_ids:
                sequences.append(sequence)
                pdb_files.append(pdb_file)
                data.append(protein)
        self.sequences = sequences
        self.pdb_files = pdb_files
        self.data = data

        splits = [os.path.basename(os.path.dirname(pdb_file)) for pdb_file in self.pdb_files]
        self.num_samples = [splits.count("train"), splits.count("valid"), splits.count("test")]
    
    
    
    def get_item(self, index):
        if getattr(self, "lazy", False):
            protein = data.Protein.from_pdb(self.pdb_files[index], self.kwargs)
        else:
            protein = self.data[index].clone()
        if hasattr(protein, "residue_feature"):
            with protein.residue():
                protein.residue_feature = protein.residue_feature.to_dense()
        item = {"graph": protein}
        if self.transform:
            item = self.transform(item)
        item["TxGNN Embeddings"] = self.embeddings[index]
        return item
    
class TargetPrediction(tasks.Task, core.Configurable):
    """
    Parameters:
        model (nn.Module): graph representation model
        task (list of int, optional): training task id(s).
        criterion (list or dict, optional): training criterion(s). For dict, the keys are criterions and the values
            are the corresponding weights. Available criterions are ``bce``.
        metric (str or list of str, optional): metric(s).
            Available metrics are ``auroc@macro``, ``auprc@macro``, ``auroc@micro``, ``auprc@micro`` and ``f1_max``.
        num_mlp_layer (int, optional): number of layers in the MLP prediction head
        normalization (bool, optional): whether to normalize the target
        reweight (bool, optional): whether to re-weight tasks according to the number of positive samples
        graph_construction_model (nn.Module, optional): graph construction model
        verbose (int, optional): output verbose level
    """

    eps = 1e-10
    _option_members = {"criterion", "metric"}

    def __init__(self, model, embedding, task=(), criterion="bce", metric=("auprc@micro", "f1_max"), num_mlp_layer=1,
                 normalization=True, reweight=False, graph_construction_model=None, verbose=0, hidden_dims=None,
                 batch_norm=False, dropout=0, activation='relu'):
        super(TargetPrediction, self).__init__()
        self.model = model
        self.task = task
        self.register_buffer("task_indices", torch.LongTensor(task))
        self.criterion = criterion
        self.metric = metric
        self.num_mlp_layer = num_mlp_layer
        self.normalization = normalization
        self.reweight = reweight
        self.graph_construction_model = graph_construction_model
        self.verbose = verbose
        self.embedding_dim = embedding

        if hidden_dims == None:
            hidden_dims = [self.model.output_dim] * (self.num_mlp_layer - 1)
        self.mlp = layers.MLP(self.model.output_dim, hidden_dims + [self.embedding_dim],
                              batch_norm=batch_norm, dropout=dropout, activation=activation)

    def preprocess(self, train_set, valid_set, test_set):
        """
        Compute the weight for each task on the training set.
        """
        values = []
        for data in train_set:
            values.append(data["TxGNN Embeddings"])
        values = torch.stack(values, dim=0)    
        
        if self.reweight:
            num_positive = values.sum(dim=0)
            weight = (num_positive.mean() / num_positive).clamp(1, 10)
        else:
            weight = torch.ones(len(self.task), dtype=torch.float)

        self.register_buffer("weight", torch.as_tensor(weight, dtype=torch.float))


    def forward(self, batch):
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred = self.predict(batch, all_loss, metric)
        target = batch['TxGNN Embeddings']

        for criterion, weight in self.criterion.items():
            if criterion == "mse":
                loss = F.mse_loss(pred, target, reduction="mean")
            elif criterion == "huber":
                loss = F.huber_loss(pred, target, reduction='mean')
            else:
                raise ValueError("Unknown criterion `%s`" % criterion)
            
            name = tasks._get_criterion_name(criterion)
            metric[name] = loss
            all_loss += loss * weight

        return all_loss, metric


    def predict(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        if self.graph_construction_model:
            graph = self.graph_construction_model(graph)
        output = self.model(graph, graph.node_feature.float(), all_loss=all_loss, metric=metric)
        pred = self.mlp(output["graph_feature"])
        return pred

    def target(self, batch):
        target = batch["TxGNN Embeddings"]
        return target

    def evaluate(self, pred, target):
        metric = {}
        for _metric in self.metric:
            if _metric == "auroc@micro":
                score = metrics.area_under_roc(pred.flatten(), target.long().flatten())
            elif _metric == "auroc@macro":
                score = metrics.variadic_area_under_roc(pred, target.long(), dim=0).mean()
            elif _metric == "auprc@micro":
                score = metrics.area_under_prc(pred.flatten(), target.long().flatten())
            elif _metric == "auprc@macro":
                score = metrics.variadic_area_under_prc(pred, target.long(), dim=0).mean()
            elif _metric == "f1_max":
                score = metrics.f1_max(pred, target)
            elif _metric == "r2":
                score = metrics.r2(pred, target)
            elif _metric == "pearson":
                score = pearsonr(pred, torch.transpose(target,0,1))
            elif _metric == "spearman":
                score = spearmanr(pred, torch.transpose(target,0,1))
            else:
                raise ValueError("Unknown criterion `%s`" % _metric)

            name = tasks._get_metric_name(_metric)
            metric[name] = score

        return metric
    
def pearsonr(pred, target):
    """
    Pearson correlation between prediction and target.

    Parameters:
        pred (Tensor): prediction of shape :math: `(N,)`
        target (Tensor): target of shape :math: `(N,)`
    """
    pred_mean = pred.float().mean()
    target_mean = target.float().mean()
    pred_centered = pred - pred_mean
    target_centered = target - target_mean
    pred_normalized = pred_centered / pred_centered.norm(2)
    target_normalized = target_centered / target_centered.norm(2)
    pearsonr = pred_normalized @ target_normalized
    return pearsonr

def spearmanr(pred, target):
    """
    Spearman correlation between prediction and target.

    Parameters:
        pred (Tensor): prediction of shape :math: `(N,)`
        target (Tensor): target of shape :math: `(N,)`
    """

    def get_ranking(input):
        input_set, input_inverse = input.unique(return_inverse=True)
        order = input_inverse.argsort()
        ranking = torch.zeros(len(input_inverse), device=input.device)
        ranking[order] = torch.arange(1, len(input) + 1, dtype=torch.float, device=input.device)

        # for elements that have the same value, replace their rankings with the mean of their rankings
        mean_ranking = scatter_mean(ranking, input_inverse, dim=0, dim_size=len(input_set))
        ranking = mean_ranking[input_inverse]
        return ranking

    pred = get_ranking(pred)
    target = get_ranking(target)
    covariance = (pred * target).mean() - pred.mean() * target.mean()
    pred_std = pred.std(unbiased=False)
    target_std = target.std(unbiased=False)
    spearmanr = covariance / (pred_std * target_std + 1e-10)
    return spearmanr
    

class Engine_WandB(core.Configurable):
    """
    Parameters:
        task (nn.Module): task
        train_set (data.Dataset): training set
        valid_set (data.Dataset): validation set
        test_set (data.Dataset): test set
        optimizer (optim.Optimizer): optimizer
        scheduler (lr_scheduler._LRScheduler, optional): scheduler
        gpus (list of int, optional): GPU ids. By default, CPUs will be used.
            For multi-node multi-process case, repeat the GPU ids for each node.
        batch_size (int, optional): batch size of a single CPU / GPU
        gradient_interval (int, optional): perform a gradient update every n batches.
            This creates an equivalent batch size of ``batch_size * gradient_interval`` for optimization.
        num_worker (int, optional): number of CPU workers per GPU
        logger (str or core.LoggerBase, optional): logger type or logger instance.
            Available types are ``logging`` and ``wandb``.
        log_interval (int, optional): log every n gradient updates
    """

    def __init__(self, task, train_set, valid_set, test_set, optimizer, scheduler=None, gpus=None, batch_size=1,
                 gradient_interval=1, num_worker=0, logger="logging", log_interval=100, wandb_project = None, wandb_name = None):
        module_var = sys.modules[__name__]

        self.rank = comm.get_rank()
        self.world_size = comm.get_world_size()
        self.gpus = gpus
        self.batch_size = batch_size
        self.gradient_interval = gradient_interval
        self.num_worker = num_worker

        if gpus is None:
            self.device = torch.device("cpu")
        else:
            if len(gpus) != self.world_size:
                error_msg = "World size is %d but found %d GPUs in the argument"
                if self.world_size == 1:
                    error_msg += ". Did you launch with `python -m torch.distributed.launch`?"
                raise ValueError(error_msg % (self.world_size, len(gpus)))
            self.device = torch.device(gpus[self.rank % len(gpus)])

        if self.world_size > 1 and not dist.is_initialized():
            #if self.rank == 0:
                #module_var.logger.info("Initializing distributed process group")
            backend = "gloo" if gpus is None else "nccl"
            comm.init_process_group(backend, init_method="env://")

        if hasattr(task, "preprocess"):
            #if self.rank == 0:
                #module_var.logger.warning("Preprocess training set")
            old_params = list(task.parameters())
            result = task.preprocess(train_set, valid_set, test_set)
            if result is not None:
                train_set, valid_set, test_set = result
            new_params = list(task.parameters())
            if len(new_params) != len(old_params):
                optimizer.add_param_group({"params": new_params[len(old_params):]})
        if self.world_size > 1:
            task = nn.SyncBatchNorm.convert_sync_batchnorm(task)
            buffers_to_ignore = []
            for name, buffer in task.named_buffers():
                if not isinstance(buffer, torch.Tensor):
                    buffers_to_ignore.append(name)
            task._ddp_params_and_buffers_to_ignore = set(buffers_to_ignore)
        if self.device.type == "cuda":
            task = task.cuda(self.device)

        self.model = task
        self.train_set = train_set
        self.valid_set = valid_set
        self.test_set = test_set
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.log_interval = log_interval

        if isinstance(logger, str):
            if logger == "logging":
                self.logger = core.LoggingLogger()
            elif logger == "wandb":
                if not wandb_project:
                    wandb_project = task.__class__.__name__
                self.logger = core.WandbLogger(project=wandb_project, name=wandb_name)
            else:
                raise ValueError("Unknown logger `%s`" % logger)
        self.meter = core.Meter(log_interval=log_interval, silent=self.rank > 0, logger=self.logger)
        self.meter.log_config(self.config_dict())
    
    def train(self, num_epoch=1, batch_per_epoch=None, save_epochs=1, save_name = "model", protgnn=False):
        """
        Train the model.

        If ``batch_per_epoch`` is specified, randomly draw a subset of the training set for each epoch.
        Otherwise, the whole training set is used for each epoch.

        Parameters:
            num_epoch (int, optional): number of epochs
            batch_per_epoch (int, optional): number of batches per epoch
        """
        sampler = torch_data.DistributedSampler(self.train_set, self.world_size, self.rank)
        dataloader = data.DataLoader(self.train_set, self.batch_size, sampler=sampler, num_workers=self.num_worker)
        valid_sampler = torch_data.DistributedSampler(self.valid_set, self.world_size, self.rank)
        valid_dataloader = data.DataLoader(self.valid_set, self.batch_size, sampler=valid_sampler, num_workers=self.num_worker)

        print(f"Number of batches per epoch: {len(dataloader)}")
        batch_per_epoch = batch_per_epoch or len(dataloader)
        model = self.model
        model.split = "train"
        if self.world_size > 1:
            if self.device.type == "cuda":
                model = nn.parallel.DistributedDataParallel(model, device_ids=[self.device],
                                                            find_unused_parameters=True)
            else:
                model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        model.train()

        for epoch in self.meter(num_epoch):
            model.train()
            model.split = "train"
            sampler.set_epoch(epoch)

            metrics = []
            start_id = 0
            epoch_loss = 0
            # the last gradient update may contain less than gradient_interval batches
            gradient_interval = min(batch_per_epoch - start_id, self.gradient_interval)
            total_batches = 0

            include_last = True
            for batch_id, batch in enumerate(islice(dataloader, batch_per_epoch)):
                if protgnn:
                    if batch['embeddings'].size()[0] != 1:
                        if self.device.type == "cuda":
                            batch = utils.cuda(batch, device=self.device)
                        loss, metric = model(batch)
                        if not loss.requires_grad:
                            raise RuntimeError("Loss doesn't require grad. Did you define any loss in the task?")
                        loss = loss / gradient_interval
                        loss.backward()
                        metrics.append(metric)
                        epoch_loss += loss.item()

                        if batch_id - start_id + 1 == gradient_interval:
                            self.optimizer.step()
                            self.optimizer.zero_grad()

                            metric = utils.stack(metrics, dim=0)
                            metric = utils.mean(metric, dim=0)
                            if self.world_size > 1:
                                metric = comm.reduce(metric, op="mean")
                            
                            if batch_id%self.log_interval == 0:
                                self.logger.log(metric, step_id=total_batches, category="Train Batch")

                            metrics = []
                            start_id = batch_id + 1
                            gradient_interval = min(batch_per_epoch - start_id, self.gradient_interval)
                        total_batches += 1

                    else:    
                        include_last = False
                else:
                    if self.device.type == "cuda":
                        batch = utils.cuda(batch, device=self.device)

                    loss, metric = model(batch)
                    if not loss.requires_grad:
                        raise RuntimeError("Loss doesn't require grad. Did you define any loss in the task?")
                    loss = loss / gradient_interval
                    loss.backward()
                    metrics.append(metric)
                    epoch_loss += loss.item()

                    if batch_id - start_id + 1 == gradient_interval:
                        self.optimizer.step()
                        self.optimizer.zero_grad()

                        metric = utils.stack(metrics, dim=0)
                        metric = utils.mean(metric, dim=0)
                        if self.world_size > 1:
                            metric = comm.reduce(metric, op="mean")
                        
                        if batch_id%self.log_interval == 0:
                            self.logger.log(metric, step_id=total_batches, category="Train Batch")

                        metrics = []
                        start_id = batch_id + 1
                        gradient_interval = min(batch_per_epoch - start_id, self.gradient_interval)
                    total_batches += 1

            self.logger.log({'loss':epoch_loss/total_batches}, step_id = epoch, category="Train Epoch")

            #validation
            model.split = "valid"
            valid_loss = 0
            include_last = True
            for batch in valid_dataloader:
                if protgnn:
                    if batch['embeddings'].size()[0] != 1:
                        if self.device.type == "cuda":
                            batch = utils.cuda(batch, device=self.device)
                        loss, metric = model(batch)
                        valid_loss += loss.item()
                        include_last = False
                else:
                    if self.device.type == "cuda":
                        batch = utils.cuda(batch, device=self.device)
                    loss, metric = model(batch)
                    valid_loss += loss.item()

            valid_metric = self.evaluate('valid', log=False)
            if include_last:
                valid_metric['loss'] = valid_loss/len(valid_dataloader)
            else:
                valid_metric['loss'] = valid_loss/(len(valid_dataloader)-1)
            valid_metric_int = {}
            for key in valid_metric:
                try:
                    valid_metric_int[key] = float(valid_metric[key])
                except:
                    pass
            self.logger.log(valid_metric_int, step_id = epoch, category="Valid Epoch")
            
            if epoch % save_epochs == 0 and epoch != 0:
                self.save(save_name + f"_epoch{epoch}.pth")

            if self.scheduler:
                self.scheduler.step()


    @torch.no_grad()
    def evaluate(self, split, log=True):
        """
        Evaluate the model.

        Parameters:
            split (str): split to evaluate. Can be ``train``, ``valid`` or ``test``.
            log (bool, optional): log metrics or not

        Returns:
            dict: metrics
        """
        #if comm.get_rank() == 0:
            #self.logger.warning(pretty.separator)
            #self.logger.warning("Evaluate on %s" % split)
        test_set = getattr(self, "%s_set" % split)
        sampler = torch_data.DistributedSampler(test_set, self.world_size, self.rank)
        dataloader = data.DataLoader(test_set, self.batch_size, sampler=sampler, num_workers=self.num_worker)
        model = self.model
        model.split = split

        model.eval()
        preds = []
        targets = []
        for batch in dataloader:
            if self.device.type == "cuda":
                batch = utils.cuda(batch, device=self.device)

            pred, target = model.predict_and_target(batch)
            preds.append(pred)
            targets.append(target)

        pred = utils.cat(preds)
        target = utils.cat(targets)
        if self.world_size > 1:
            pred = comm.cat(pred)
            target = comm.cat(target)
        metric = model.evaluate(pred, target)
        if log:
            self.meter.log(metric, category="%s/epoch" % split)

        return metric
    
    @torch.no_grad()
    def retrieve_embeddings(self, dataset, split='test'):
        test_set = getattr(self, "%s_set" % split)
        dataloader = data.DataLoader(dataset, 1, num_workers=self.num_worker)
        model = self.model
        model.split = split

        model.eval()
        preds = []
        targets = []
        for batch in dataloader:
            if self.device.type == "cuda":
                batch = utils.cuda(batch, device=self.device)

            pred, target = model.predict_and_target(batch)
            preds.append(pred)
            targets.append(target)

        return preds, targets


    def load(self, checkpoint, load_optimizer=True, strict=True, model_only=True, remove_mlp = True):
        """
        Load a checkpoint from file.

        Parameters:
            checkpoint (file-like): checkpoint file
            load_optimizer (bool, optional): load optimizer state or not
            strict (bool, optional): whether to strictly check the checkpoint matches the model parameters
        """
        #if comm.get_rank() == 0:
            #self.logger.warning("Load checkpoint from %s" % checkpoint)
        checkpoint = os.path.expanduser(checkpoint)
        state = torch.load(checkpoint, map_location=self.device)

        if model_only:
            state_model = {'model.'+k:v for k,v in state.items()}
        else:
            state_model = state['model']
        
        if remove_mlp:
            state_model = {k: v for k, v in state_model.items() if 'mlp' not in k}
            
        if not strict:
            state_model = {k: v for k, v in state_model.items() if k != 'task_indices'}
            state_model = {k: v for k, v in state_model.items() if k != 'weight'}
        

        self.model.load_state_dict(state_model, strict=strict)

        if load_optimizer:
            self.optimizer.load_state_dict(state["optimizer"])
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(self.device)

        comm.synchronize()


    def save(self, checkpoint):
        """
        Save checkpoint to file.

        Parameters:
            checkpoint (file-like): checkpoint file
        """
        #if comm.get_rank() == 0:
            #self.logger.warning("Save checkpoint to %s" % checkpoint)
        checkpoint = os.path.expanduser(checkpoint)
        if self.rank == 0:
            state = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict()
            }
            torch.save(state, checkpoint)

        comm.synchronize()


    @classmethod
    def load_config_dict(cls, config):
        """
        Construct an instance from the configuration dict.
        """
        if getattr(cls, "_registry_key", cls.__name__) != config["class"]:
            raise ValueError("Expect config class to be `%s`, but found `%s`" % (cls.__name__, config["class"]))

        optimizer_config = config.pop("optimizer")
        new_config = {}
        for k, v in config.items():
            if isinstance(v, dict) and "class" in v:
                v = core.Configurable.load_config_dict(v)
            if k != "class":
                new_config[k] = v
        optimizer_config["params"] = new_config["task"].parameters()
        new_config["optimizer"] = core.Configurable.load_config_dict(optimizer_config)

        return cls(**new_config)


    @property
    def epoch(self):
        """Current epoch."""
        return self.meter.epoch_id
    
class MultipleBinaryClassification(tasks.Task, core.Configurable):
    """
    Multiple binary classification task for graphs / molecules / proteins.

    Parameters:
        model (nn.Module): graph representation model
        task (list of int, optional): training task id(s).
        criterion (list or dict, optional): training criterion(s). For dict, the keys are criterions and the values
            are the corresponding weights. Available criterions are ``bce``.
        metric (str or list of str, optional): metric(s).
            Available metrics are ``auroc@macro``, ``auprc@macro``, ``auroc@micro``, ``auprc@micro`` and ``f1_max``.
        num_mlp_layer (int, optional): number of layers in the MLP prediction head
        normalization (bool, optional): whether to normalize the target
        reweight (bool, optional): whether to re-weight tasks according to the number of positive samples
        graph_construction_model (nn.Module, optional): graph construction model
        verbose (int, optional): output verbose level
    """

    eps = 1e-10
    _option_members = {"criterion", "metric"}

    def __init__(self, model, task=(), criterion="bce", metric=("auprc@micro", "f1_max"), num_mlp_layer=1,
                 normalization=True, reweight=False, graph_construction_model=None, verbose=0, batch_norm=False, dropout=0, gearnet_protgnn=False, embedding_dim = 1280):
        super(MultipleBinaryClassification, self).__init__()
        self.model = model
        self.task = task
        self.register_buffer("task_indices", torch.LongTensor(task))
        self.criterion = criterion
        self.metric = metric
        self.num_mlp_layer = num_mlp_layer
        self.normalization = normalization
        self.reweight = reweight
        self.graph_construction_model = graph_construction_model
        self.verbose = verbose

        if gearnet_protgnn:
            hidden_dims = [self.model.output_dim] * 4 
            hidden_dims = hidden_dims + [embedding_dim] *  self.num_mlp_layer
        else:
            hidden_dims = [self.model.output_dim] * (self.num_mlp_layer - 1)
        self.mlp = layers.MLP(self.model.output_dim, hidden_dims + [len(task)], batch_norm=batch_norm, dropout=dropout)

    def preprocess(self, train_set, valid_set, test_set):
        """
        Compute the weight for each task on the training set.
        """
        values = []
        for data in train_set:
            values.append(data["targets"][self.task_indices])
        values = torch.stack(values, dim=0)    
        
        if self.reweight:
            num_positive = values.sum(dim=0)
            weight = (num_positive.mean() / num_positive).clamp(1, 10)
        else:
            weight = torch.ones(len(self.task), dtype=torch.float)

        self.register_buffer("weight", torch.as_tensor(weight, dtype=torch.float))


    def forward(self, batch):
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred = self.predict(batch, all_loss, metric)
        target = self.target(batch)

        for criterion, weight in self.criterion.items():
            if criterion == "bce":
                loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
            else:
                raise ValueError("Unknown criterion `%s`" % criterion)
            loss = loss.mean(dim=0)
            loss = (loss * self.weight).sum() / self.weight.sum()
            
            name = tasks._get_criterion_name(criterion)
            metric[name] = loss
            all_loss += loss * weight

        return all_loss, metric


    def predict(self, batch, all_loss=None, metric=None):
        graph = batch["graph"]
        if self.graph_construction_model:
            graph = self.graph_construction_model(graph)
        output = self.model(graph, graph.node_feature.float(), all_loss=all_loss, metric=metric)
        pred = self.mlp(output["graph_feature"])
        return pred

    def target(self, batch):
        target = batch["targets"][:, self.task_indices]
        return target

    def evaluate(self, pred, target):
        metric = {}
        for _metric in self.metric:
            if _metric == "auroc@micro":
                score = metrics.area_under_roc(pred.flatten(), target.long().flatten())
            elif _metric == "auroc@macro":
                score = metrics.variadic_area_under_roc(pred, target.long(), dim=0).mean()
            elif _metric == "auprc@micro":
                score = metrics.area_under_prc(pred.flatten(), target.long().flatten())
            elif _metric == "auprc@macro":
                score = metrics.variadic_area_under_prc(pred, target.long(), dim=0).mean()
            elif _metric == "f1_max":
                score = metrics.f1_max(pred, target)
            else:
                raise ValueError("Unknown criterion `%s`" % _metric)

            name = tasks._get_metric_name(_metric)
            metric[name] = score

        return metric

def sweep_run():
    with wandb.init() as run:
        config = run.config

        dynamic_wandb_name = f"{wandb_name}_lr{config.learning_rate}_nl{config.num_layers}_bs{config.batch_size}_ep{config.epochs}"

        # Modify the model initialization with dynamic hyperparameters
        graph_construction_model = layers.GraphConstruction(node_layers=[geometry.AlphaCarbonNode()], 
                                                    edge_layers=[geometry.SpatialEdge(radius=10.0, min_distance=5),
                                                                 geometry.KNNEdge(k=10, min_distance=5),
                                                                 geometry.SequentialEdge(max_distance=2)],
                                                    edge_feature="gearnet")
        gearnet_edge = models.GearNet(input_dim=21, hidden_dims=[512, 512, 512, 512, 512, 512], 
                                    num_relation=7, edge_input_dim=59, num_angle_bin=8,
                                    batch_norm=True, concat_hidden=True, short_cut=True, readout="sum")

        # Assuming the rest of your setup stays the same
        
        embedding = len(dataset[0]['TxGNN Embeddings'])
        task = TargetPrediction(gearnet_edge, embedding, graph_construction_model=graph_construction_model, num_mlp_layer=config.num_layers, 
                                task=[_ for _ in range(len(dataset.tasks))], criterion="huber", metric=["auprc@micro", "r2"])
        optimizer = torch.optim.Adam(task.parameters(), lr=5e-4)
        module = sys.modules[__name__]
        logger = logging.getLogger(__name__)
        # Initialize the training engine
        solver = Engine_WandB(task, train_set, valid_set, test_set, optimizer,
                              gpus=[0], batch_size=config.batch_size, logger='wandb', log_interval=100, 
                              wandb_name=dynamic_wandb_name, wandb_project='GearNet-ProtGNN_sweep')

        # Load model, train, and save
        solver.load("models/mc_gearnet_edge.pth", load_optimizer=False, strict=False)
        solver.train(num_epoch=config.epochs, save_name=f"models/model_truncate_{config.epochs}", save_epochs=300)

