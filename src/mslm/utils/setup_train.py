import os
import torch
import random

torch.manual_seed(23)
random.seed(23)

from torch.utils.data import DataLoader, random_split

#Imported Classes
from src.mslm.models import Imitator
from src.mslm.training import Trainer
from src.mslm.dataloader import KeypointDataset, collate_fn, GRPCDataset, BatchSampler
from src.mslm.utils.paths import path_vars

#Profilers
from torch.profiler import profile, ProfilerActivity, record_function
import datetime

def setup_paths():
    """Define y retorna las rutas necesarias para datos y modelos."""
    data_path = path_vars.data_path
    model_path = path_vars.model_path
    h5_file = path_vars.h5_file
    return data_path, model_path, h5_file

def prepare_datasets(h5File, train_ratio, n_keypoints=112):
    """Carga el dataset base, lo envuelve y lo divide en entrenamiento y validación."""
    keypoint_reader = KeypointDataset(h5Path=h5File, return_label=False, n_keypoints=n_keypoints, data_augmentation=True, max_length=4000)
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

def build_model(input_size, output_size, **kwargs):
    """Construye, compila y retorna el modelo Imitator."""
    model = Imitator(input_size=input_size, output_size=output_size, **kwargs)
    print(model)
    print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f} M parameters")
    return model

def run_training(params, train_dataloader, val_dataloader, model):
    """Configura y ejecuta el entrenamiento."""
    trainer = Trainer(model, train_dataloader, val_dataloader, **params)
    trainer.ckpt_mgr.save_params(params)

    print("Starting training...")
    return trainer.train()

def profile_training(params, train_dataloader, val_dataloader, model, profile_mode: str):
    trainer = Trainer(model, train_dataloader, val_dataloader, **params)
    trainer.ckpt_mgr.save_params(params)

    if profile_mode == "nvidia":
        print("Starting training with profiling nvidia...")
        return trainer.train(prof=True)
    elif profile_mode == "pytorch_model":
        print("Starting training with profiling pytorch...")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    with_stack=True) as p:
            with record_function("train"):
                trainer.train()
            file_path = f"{path_vars.report_path}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"Saved at: {file_path}")    
            p.export_chrome_trace(f"{file_path}.json.gz")
            p.export_stacks(f"{file_path}_stacks.txt", "self_cpu_time_total")
    elif profile_mode == "pytorch_memory":
        print("Starting training with profiling memory torch...")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    with_stack=True,
                    profile_memory=True) as p:
            with record_function("train"):
                trainer.train()
            file_path = f"{path_vars.report_path}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            print(f"Saved at: {file_path}")    
            p.export_chrome_trace(f"{file_path}.json.gz")
            p.export_stacks(f"{file_path}_stacks.txt", "self_cpu_time_total")
            p.export_memory_profile(f"{file_path}_memory.txt")

    else:
        raise ValueError("Unsupported profiling mode. Use 'nvidia' or 'pytorch'.")

def run_dt_training(params, train_dataloader, val_dataloader, model, rank, channel, dist, stub):
    """Configura y ejecuta el entrenamiento."""
    trainer = Trainer(model, train_dataloader, val_dataloader, **params)
    trainer.ckpt_mgr.save_params(params)

    print("Starting training...")
    return trainer.train_dist(rank, channel, dist, stub)