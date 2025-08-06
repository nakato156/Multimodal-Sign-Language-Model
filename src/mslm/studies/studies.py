from ..training import Trainer
from ..models import Imitator
from optuna.exceptions import TrialPruned
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import trange
import torch._dynamo
import gc
import numpy as np

from src.mslm.utils.early_stopping import EarlyStopping
from src.mslm.utils.paths import path_vars


def lr_objetive(trial, train_dataloader, val_dataloader, **params):
    learning_rate = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    params["learning_rate"] = learning_rate
    modelParameters = params["modelParameters"]

    model = Imitator(
        input_size=modelParameters["input_size"],
        T_size=modelParameters["frame_clips"],
        output_size=modelParameters["output_size"],
        nhead=32,
        ff_dim=4096,
        n_layers=8,
        pool_dim=128
    ).to(modelParameters["device"])

    trainer = Trainer(model, train_dataloader, val_dataloader, **params)
    _, val_loss = trainer.train()
    return val_loss

def complete_objective(trial, train_dataloader, val_dataloader, model_params, train_config):
    adjacency_matrix = np.load(path_vars.A_matrix, allow_pickle=True)

    encoder_hidden_size   = trial.suggest_int("encoder_hidden_size", 256, 1024, step=256)
    decoder_hidden_size   = trial.suggest_int("decoder_hidden_size", 512, 1024, step=256)
    nhead         = trial.suggest_categorical("nhead", [2, 4, 8])
    ff_dim        = trial.suggest_int("ff_dim", 1024, 3072, step=256)

    learning_rate = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    n_encoder_layers        = trial.suggest_int("n_encoder_layers", 2, 8, step=2)
    n_decoder_layers        = trial.suggest_int("n_decoder_layers", 2, 8, step=2)
    encoder_dropout = trial.suggest_float("encoder_dropout", 0.1, 0.6, step=0.05)
    decoder_dropout = trial.suggest_float("decoder_dropout", 0.1, 0.6, step=0.05)

    weight_decay  = trial.suggest_float("weight_decay", 0.0, 0.1, step=0.01)
    grad_clip    = trial.suggest_float("grad_clip", 0.1, 5.0, step=0.1)


    print(f"Encoder Hidden Size: {encoder_hidden_size}, Decoder Hidden Size: {decoder_hidden_size}, Nhead: {nhead}, FF Dim: {ff_dim}, Encoder Layers: {n_encoder_layers}, Decoder Layers: {n_decoder_layers}, Learning Rate: {learning_rate}")
    print(f"Encoder Dropout: {encoder_dropout}, Decoder Dropout: {decoder_dropout}")
    train_config["learning_rate"] = learning_rate
    train_config["grad_clip"] = grad_clip

    early_stopping = EarlyStopping(patience=10)
    trainer.early_stopping = early_stopping

    model = Imitator(
        A = adjacency_matrix,
        input_size=model_params["input_size"],
        output_size=model_params["output_size"],
        encoder_hidden_size = encoder_hidden_size, 
        decoder_hidden_size = decoder_hidden_size,
        nhead=nhead,
        ff_dim=ff_dim,
        n_encoder_layers = n_encoder_layers,
        n_decoder_layers = n_decoder_layers,
        max_seq_length=20,
        encoder_dropout = encoder_dropout, 
        decoder_dropout = decoder_dropout
    )

    trainer = Trainer(model, train_dataloader, val_dataloader,save_tb_model=False , **train_config)

    trainer.optimizer = AdamW(
        trainer.model.parameters(), 
        lr=trainer.learning_rate, 
        weight_decay=weight_decay,
        foreach=True
    )

    def linear_warmup_cosine_decay(current_step, warmup_steps, total_steps):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 0.5 * (1.0 + torch.cos(
            torch.tensor((current_step - warmup_steps) / (total_steps - warmup_steps) * 3.1415926535))
        ).item()

    warmup_steps = 5 * len(trainer.train_loader)  # p.ej. 5 epochs de warm-up
    total_steps = trainer.epochs * len(trainer.train_loader)

    lr_lambda = lambda step: linear_warmup_cosine_decay(step, warmup_steps, total_steps)
    trainer.scheduler = LambdaLR(trainer.optimizer, lr_lambda=lr_lambda)
    trainer.prepare_trainer()

    for epoch in trange(trainer.epochs, desc="Epochs"):
        train_loss = trainer._train_epoch(epoch)
        val_loss   = trainer._val(epoch)
        trainer.scheduler.step()
        torch.cuda.empty_cache()

        # Reportar y podar
        trial.report(val_loss, step=epoch)
        if trial.should_prune():
            raise TrialPruned(f"Pruned at epoch {epoch} with val_loss {val_loss:.4f}")

        # Early stopping interno
        if trainer.early_stopping.stop:
            break
    
    torch._dynamo.reset()
    try:
        del model, trainer
    except NameError:
        pass
    gc.collect()
    torch.cuda.empty_cache()
    return val_loss