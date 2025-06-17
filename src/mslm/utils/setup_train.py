import torch

from torch.utils.data import DataLoader, random_split

#Imported Classes
from src.mslm.models import Imitator
from src.mslm.training import Trainer
from src.mslm.dataloader import KeypointDataset, collate_fn
from src.mslm.utils.paths import path_vars

#Profilers
from torch.profiler import profile, ProfilerActivity
import os
import datetime

def setup_paths():
    """Define y retorna las rutas necesarias para datos y modelos."""
    data_path = path_vars.data_path
    model_path = path_vars.model_path
    h5_file = path_vars.h5_file
    return data_path, model_path, h5_file

def prepare_datasets(h5File, train_ratio, device):
    """Carga el dataset base, lo envuelve y lo divide en entrenamiento y validación."""
    keypoint_reader = KeypointDataset(h5Path=h5File, return_label=False)

    train_dataset, validation_dataset = random_split(keypoint_reader, [train_ratio, 1 - train_ratio], generator=torch.Generator().manual_seed(42))
    print(f"Train size:\t{len(train_dataset)}\nValidation size:\t{len(validation_dataset)}")
    return train_dataset, validation_dataset

def create_dataloaders(train_dataset, validation_dataset, batch_size, num_workers=4):
    """Crea y retorna los DataLoaders para entrenamiento y validación."""
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn
    )
    val_dataloader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        collate_fn=collate_fn
    )
    return train_dataloader, val_dataloader

def build_model(input_size, output_size, device, compile=True, **kwargs):
    """Construye, compila y retorna el modelo Imitator."""
    model = Imitator(input_size=input_size, output_size=output_size, **kwargs).to(device)
    if compile:
        model = torch.compile(model, backend="inductor", mode="reduce-overhead")
    print(model)
    print(f"{sum(p.numel() for p in model.parameters())/1e6:.2f} M parameters")
    return model

def run_training(params, train_dataloader, val_dataloader, model, profile_pytorch=False):
    """Configura y ejecuta el entrenamiento."""
    trainer = Trainer(model, train_dataloader, val_dataloader, **params)
    trainer.ckpt_mgr.save_params(params)

    if profile_pytorch:
        print("Starting training with profiling...")
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                    record_shapes=True,
                    with_stack=True,
                    profile_memory=True) as p:
            trainer.train()
        p.export_memory_timeline(f"../outputs/profile/{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")    
    else:
        print("Starting training...")
        return trainer.train()