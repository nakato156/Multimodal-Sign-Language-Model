import optuna
import torch
from src.mslm.utils.setup_train import setup_paths, prepare_datasets, create_dataloaders
from src.mslm.studies import complete_objective
from src.mslm.utils import ConfigLoader

def run(
    n_trials: int = 10,
    batch_size: int = 32,
    train_ratio: float = 0.8,
):
    # setup
    _, _, h5_file = setup_paths()
        
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_parameters = ConfigLoader("config/model/config.toml").load_config()
    model_parameters.update({
        "device": device if model_parameters.get("device") == "auto" else model_parameters.get("device", device),
        "input_size": 250 * 2,
        "output_size": 3072,
    })

    # --- config de entrenamiento ---
    train_config = ConfigLoader("config/training/train_config.toml").load_config()
    train_ratio = train_config.get("train_ratio", train_ratio)
    train_config.update({
        "epochs": n_trials,
        "batch_size": batch_size if batch_size else train_config.get("batch_size", 32),
        "checkpoint_interval": train_config.get("checkpoint_interval", 5),
        "log_interval": train_config.get("log_interval", 2),
        "train_ratio": train_ratio,
        "validation_ratio": round(1 - train_ratio, 2),
        "device": device if model_parameters.get("device") == "auto" else model_parameters.get("device", device),
    })

    # datasets
    tr_ds, val_ds = prepare_datasets(h5_file, train_ratio)
    tr_dl, val_dl = create_dataloaders(tr_ds, val_ds, batch_size, num_workers=4)

    # optuna
    storage = f"sqlite:///new_study.db"
    study = optuna.create_study(study_name="new_study",
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
    parser.add_argument("--n_trials", type=int, default=8, help="Number of trials to run.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--train_ratio", type=float, default=0.8, help="Ratio of training data.")
    args = parser.parse_args()

    run(args.n_trials, args.batch_size, args.train_ratio)