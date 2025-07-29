import os
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import optuna
import torch
import random

torch.manual_seed(23)
random.seed(23)

from src.mslm.utils.setup_train import setup_paths, prepare_datasets, create_dataloaders
from src.mslm.studies import complete_objective
from src.mslm.utils import ConfigLoader

def run(
    epochs: int = 15,
    n_trials: int = 10,
    batch_size: int = 32,
    batch_sample: int = 8,
    train_ratio: float = 0.8,
    batch_sampling: bool = True
):
    # setup
    _, _, h5_file = setup_paths()
        
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_parameters = ConfigLoader("config/model/config.toml").load_config()
    model_parameters.update({
        "input_size": 133 * 2,
        "output_size": 3072,
    })

    # --- config de entrenamiento ---
    train_config = ConfigLoader("config/training/train_config.toml").load_config()
    train_ratio = train_config.get("train_ratio", train_ratio)
    train_config.update({
        "learning_rate": train_config.get("learning_rate", 0.00238),
        "epochs": epochs if epochs else train_config.get("epochs", 100),
        "batch_size": batch_size if batch_size else train_config.get("batch_size", 32),
        "batch_sample": batch_sample if batch_sample else train_config.get("sub_batch_size", 32),
        "train_ratio": train_ratio,
        "validation_ratio": round(1 - train_ratio, 2),
        "device": device if model_parameters.get("device") == "auto" else model_parameters.get("device", device),
    })

    train_config["batch_sampling"] = batch_sampling
    train_config["batch_sample"] = batch_sample
    train_config["compile"] = True

    if batch_sampling:
        if batch_size%batch_sample != 0 or batch_size < batch_sample:
            raise ValueError(f"The sub_batch {batch_sample} needs to be divisible the batch size {batch_size}")

    print(f"Running study with batch size {batch_size}, sub batch size {batch_sample}")

    # datasets
    tr_ds, val_ds, tr_len, val_len = prepare_datasets(h5_file, train_ratio, 133)
    tr_dl, val_dl = create_dataloaders(tr_ds, val_ds, batch_size, num_workers=6, train_length=tr_len, val_length=val_len)

    print("Batch Size: ", train_config["batch_size"])
    # optuna
    storage = f"sqlite:///study_models.db"
    study = optuna.create_study(study_name=f"model_{train_config['model_version']}_{train_config['checkpoint']}",
        storage=storage,
        direction="minimize",
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5)
    )
    study.optimize(
        lambda t: complete_objective(t,
            model_params=model_parameters,
            train_dataloader=tr_dl,
            val_dataloader=val_dl,
            train_config=train_config,
        ),
        n_trials=n_trials
    )
    print(study.best_trial)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train.")
    parser.add_argument("--n_trials", type=int, default=8, help="Number of trials to run.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--batch_sample", type=int, default=8, help="Batch size for training.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of training data.")
    parser.add_argument("--batch_sampling", type=bool, default=True, help="Enables batch sampling for training.")
    args = parser.parse_args()

    print(f"Running study with {args.n_trials} trials, batch size {args.batch_size}, train ratio {args.train_ratio}")
    run(args.epochs, args.n_trials, args.batch_size, args.batch_sample, args.train_ratio, args.batch_sampling)