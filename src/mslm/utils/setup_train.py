import torch
import random

torch.manual_seed(23)
random.seed(23)
torch.set_default_dtype(torch.float32) 

from torch.utils.data import DataLoader, random_split

#Imported Classes
from src.mslm.models import Imitator
from src.mslm.training import Trainer
from src.mslm.dataloader import KeypointDataset, collate_fn, GRPCDataset, BatchSampler
from src.mslm.utils.paths import path_vars

#Profilers
from torch.profiler import profile, ProfilerActivity
import datetime

import torch._dynamo as dt
#dt.config.cache_size_limit = 8192
#dt.config.suppress_errors = True

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32   = True
torch.set_default_dtype(torch.float32)

#torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True

def setup_paths():
    """Define y retorna las rutas necesarias para datos y modelos."""
    data_path = path_vars.data_path
    model_path = path_vars.model_path
    h5_file = path_vars.h5_file
    return data_path, model_path, h5_file

def prepare_datasets(h5File, train_ratio, n_keypoints=245):
    """Carga el dataset base, lo envuelve y lo divide en entrenamiento y validación."""
    keypoint_reader = KeypointDataset(h5Path=h5File, return_label=False, n_keypoints=n_keypoints, data_augmentation=True)
    train_dataset, validation_dataset, train_length, val_length = keypoint_reader.split_dataset(train_ratio)

    print(f"Train size:\t{len(train_dataset)}\nValidation size:\t{len(validation_dataset)}")
    return train_dataset, validation_dataset, train_length, val_length

def create_dataloaders(train_dataset, validation_dataset, batch_size, num_workers=4, use_grpc=False, grpc_address=None, rank = 4, world_size = 4, train_length = None, val_length = None):
    """Crea y retorna los DataLoaders para entrenamiento y validación."""
    if use_grpc:
        gtrain_dataset = GRPCDataset(grpc_address, rank, world_size, split="train")
        gval_dataset = GRPCDataset(grpc_address, rank, world_size, split="val")
        
        batch_per_worker = int(batch_size/world_size)

        train_dataloader = DataLoader(gtrain_dataset,
        batch_size=batch_per_worker,           
        num_workers=4,           
        pin_memory=True)

        val_dataloader = DataLoader(gval_dataset,
        batch_size=batch_per_worker,           
        num_workers=4,           
        pin_memory=True)

    else:
        train_sampler = BatchSampler(train_length, batch_size)
        val_sampler = BatchSampler(val_length, batch_size)

        train_dataloader = DataLoader(
            train_dataset,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=collate_fn,
            batch_sampler=train_sampler
        )
        val_dataloader = DataLoader(
            validation_dataset,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=collate_fn,
            batch_sampler=val_sampler
        )
    return train_dataloader, val_dataloader

def build_model(input_size, output_size, device, compile=True, **kwargs):
    """Construye, compila y retorna el modelo Imitator."""
    model = Imitator(input_size=input_size, output_size=output_size, **kwargs)
    model = model.to(torch.float)
    model = model.to(device)
    if compile:
        model = torch.compile(model,
                              dynamic=True
        )
    print(model)
    print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f} M parameters")
    return model

def run_training(params, train_dataloader, val_dataloader, model, profile_model=0):
    """Configura y ejecuta el entrenamiento."""
    trainer = Trainer(model, train_dataloader, val_dataloader, **params)
    trainer.ckpt_mgr.save_params(params)

    if profile_model == 1:
        return trainer.train(prof=True)
    elif profile_model == 2:
        print("Starting training with profiling...")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    with_stack=True,
                    profile_memory=True) as p:
            trainer.train()
        file_path = f"{path_vars.report_path}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"Saved at: {file_path}")    
        p.export_chrome_trace(f"{file_path}.json.gz")
        p.export_memory_timeline(f"{file_path}.html", device="cuda:0")
    else:
        print("Starting training...")
        return trainer.train()

def run_dt_training(params, train_dataloader, val_dataloader, model, rank, channel, dist, stub):
    """Configura y ejecuta el entrenamiento."""
    trainer = Trainer(model, train_dataloader, val_dataloader, **params)
    trainer.ckpt_mgr.save_params(params)

    print("Starting training...")
    return trainer.train_dist(rank, channel, dist, stub)