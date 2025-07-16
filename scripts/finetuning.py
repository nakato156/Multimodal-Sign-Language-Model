import os
from pathlib import Path
import gc

from src.mslm.utils.setup_train import *
from src.mslm.utils import create_dataloaders, prepare_datasets, ConfigLoader
from src.mslm.checkpoint.manager import CheckpointManager
from src.mslm.models.components.positional_encoding import PositionalEncoding
from src.mslm.utils.early_stopping import EarlyStopping

from unsloth import FastLanguageModel

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import autocast # , GradScaler
import torch.nn.functional as F
from tqdm import tqdm

LOG = False

llama_model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-3B-Instruct",
    max_seq_length=4096,
    dtype=None,
    load_in_4bit=True,
)

torch.serialization.add_safe_globals([Imitator, PositionalEncoding])
checkpoint_name = "../outputs/checkpoints/41/1/5/model.pt"

def train(mslm, lm_head, train_dataloader, val_dataloader, llama_embed_layer, epochs, learning_rate=5e-3, 
          checkpoint_interval=2, checkpoint_manager:CheckpointManager=None, early_stopping=None, **training_args):

    device = training_args.get("device")

    # scaler = GradScaler(device=device)
    criterion = nn.KLDivLoss(reduction="batchmean")
    optimizer = torch.optim.AdamW(mslm.parameters(), lr=learning_rate)
    
    total_steps  = epochs * len(train_dataloader)
    warmup_steps = int(0.1 * total_steps)
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps - warmup_steps,
        eta_min=learning_rate * 1e-2
    )

    for epoch in tqdm(range(epochs), desc="Training Epochs", colour="green"):
        epoch_loss = 0.0
        mslm.train()
        for keypoints, input_ids in train_dataloader:
            keypoints = keypoints.to(device, non_blocking=True)
            input_ids = input_ids.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast(device_type=device, dtype=torch.bfloat16):
                pred_embed = mslm(keypoints).to(dtype=torch.bfloat16)
                pred_logits = lm_head(pred_embed)
                B, T, V = pred_logits.shape
                pred_log_probs = F.log_softmax(pred_logits, dim=-1)

                with torch.no_grad():
                    embeddings = llama_embed_layer(input_ids)
                    true_logits = lm_head(embeddings)
                    true_probs = F.softmax(true_logits, dim=-1)
                    
                    pred_log_probs = pred_log_probs.view(B*T, V)
                    true_probs = true_probs.view(B*T, V)
                
                loss = criterion(
                    pred_log_probs,
                    true_probs
                )
                
                epoch_loss += loss.detach()
                loss.backward()
                
                optimizer.step()
                scheduler.step()
        
        epoch_loss /= len(train_dataloader)
        
        # Validation loss
        val_loss = 0.0
        mslm.eval()
        with torch.no_grad():
            for keypoints, input_ids in val_dataloader:
                keypoints = keypoints.to(device, non_blocking=True)
                input_ids = input_ids.to(device, non_blocking=True)
                embeddings = llama_embed_layer(input_ids)

                pred_embed = mslm(keypoints).to(dtype=torch.bfloat16, non_blocking=True)
                pred_logits = lm_head(pred_embed)
                pred_log_probs = F.log_softmax(pred_logits, dim=-1)
                B, T, V = pred_logits.shape

                with torch.no_grad():
                    true_logits = lm_head(embeddings)
                    true_log_probs = F.softmax(true_logits, dim=-1)

                    pred_log_probs = pred_log_probs.view(B*T, V)
                    true_log_probs = true_log_probs.view(B*T, V)
                
                loss = criterion(
                    pred_log_probs,
                    true_log_probs
                ).detach()
                val_loss += loss
        val_loss /= len(val_dataloader)
        # scheduler.step(val_loss)
        
        if epoch % 2 == 0:
            tqdm.write(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}\tValidation Loss: {val_loss:.4f}")

        # Early stopping
        if early_stopping is not None:
            early_stopping(val_loss.item())
            if early_stopping.stop:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # Periodic checkpoint saving
        if checkpoint_manager and (epoch + 1) % checkpoint_interval == 0:
            checkpoint_manager.save_checkpoint(mslm, epoch + 1)

def main(model_parameters, mslm, llama_embed_layer, lm_head, **training_args):
    # Paths
    data_path, model_path, h5_file, csv_file = setup_paths()
    device = model_parameters["device"]
    train_ratio = training_args.get("train_ratio", 0.8)
    train_dataset, validation_dataset = prepare_datasets(h5_file, csv_file, tokenizer,
                                     model_parameters["T_size"],
                                     train_ratio, device)

    # Create DataLoaders
    batch_size = training_args.get("batch_size", 32)
    train_dataloader, val_dataloader = create_dataloaders(
        train_dataset, validation_dataset, batch_size
    )

    # Early stopping and checkpoint setup
    checkpoint_dir = Path(__file__).parent / "outputs" / "checkpoints" / "finetuning"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    #checkpoint manager
    checkpoint_manager = CheckpointManager(checkpoint_dir, training_args.get("model_version", 0), training_args.get("checkpoint", 0))

    early_stopping = EarlyStopping(patience=5, threshold=0.002, verbose=True)

    while True:
        torch.cuda.empty_cache()
        if not gc.collect():
            break
    
    train(
        mslm, lm_head, train_dataloader, val_dataloader,
        llama_embed_layer=llama_embed_layer,
        early_stopping=early_stopping,
        checkpoint_manager=checkpoint_manager,
        **training_args
    )

def run(epochs=100, batch_size=32, checkpoint_interval=5, log_interval=2):
    global llama_model
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ModelPath = Path(os.path.join(os.getcwd(), "model"))
    mslm = torch.load(checkpoint_name, weights_only=False).to(device)
    
    llama_embed_layer = llama_model.get_input_embeddings()
    lm_head = llama_model.lm_head.to("cuda")
    for param in lm_head.parameters():
        param.requires_grad = False

    del llama_model

    while True:
        torch.cuda.empty_cache()
        if not gc.collect():
            break

    model_parameters = ConfigLoader("config/model/config.toml").load_config()
    model_parameters.update({
        "device": device if model_parameters.get("device") == "auto" else model_parameters.get("device", device),
        "input_size": 543 * 2,
        "output_size": 3072,
        "T_size": 15 * 35,
    })

    # --- config de entrenamiento ---
    train_config = ConfigLoader("config/training/finetuning_config.toml").load_config()
    train_ratio = train_config.get("train_ratio", 0.8)
    train_config.update({
        "checkpoint": 1,
        "learning_rate": train_config.get("learning_rate", 5e-3),
        "epochs": epochs if epochs else train_config.get("epochs", 100),
        "batch_size": batch_size if batch_size else train_config.get("batch_size", 32),
        "checkpoint_interval": checkpoint_interval if checkpoint_interval else train_config.get("checkpoint_interval", 5),
        "log_interval": log_interval if log_interval else train_config.get("log_interval", 2),
        "train_ratio": train_ratio,
        "validation_ratio": round(1 - train_ratio, 2),
        "device": device if model_parameters.get("device") == "auto" else model_parameters.get("device", device),
    })

    try:
        main(model_parameters, mslm, llama_embed_layer, lm_head, **train_config)
    finally:
        folder = ModelPath / "checkpoints" / str(train_config["model_version"]) / str(train_config["checkpoint"]) 
        if not folder.exists():
            folder.mkdir(parents=True)
        torch.save(mslm, folder / "final_model.pt")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--checkpoint_interval", type=int, default=5, help="Interval for saving checkpoints.")
    parser.add_argument("--log_interval", type=int, default=2, help="Interval for logging training progress.")
    args = parser.parse_args()

    run(args.epochs, args.batch_size, args.checkpoint_interval, args.log_interval)