from ..training import Trainer
from ..models import Imitator
from optuna.exceptions import TrialPruned
import torch
from torch.optim import AdamW

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
    hidden_size   = trial.suggest_categorical("hidden_size", [512, 1024])
    nhead         = trial.suggest_categorical("nhead",       [4, 8, 16, 32, 64])
    ff_dim        = trial.suggest_int("ff_dim", 1024, 3072, step=256)
    n_layers      = trial.suggest_categorical("n_layers",    [2, 4, 6, 8, 10, 12])
    learning_rate = trial.suggest_float("lr", 1e-8, 1e-1, log=True)

    train_config["learning_rate"] = learning_rate

    model = Imitator(
        input_size=model_params["input_size"],
        output_size=model_params["output_size"],
        hidden_size=hidden_size,
        nhead=nhead,
        ff_dim=ff_dim,
        n_layers=n_layers,
        max_seq_length=301
    ).to(model_params["device"])

    trainer = Trainer(model, train_dataloader, val_dataloader, **train_config)

    optimizer = AdamW(trainer.model.parameters(), lr=trainer.learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-7
    )

    for epoch in range(trainer.epochs):
        _ = trainer._train_epoch(epoch, optimizer, scheduler=None)
        val_loss   = trainer._validate(epoch)
        scheduler.step(val_loss)

        # Reportar y podar
        trial.report(val_loss, step=epoch)
        if trial.should_prune():
            raise TrialPruned(f"Pruned at epoch {epoch} with val_loss {val_loss:.4f}")

        # Early stopping interno
        if trainer.early_stopping.stop:
            break

    return val_loss