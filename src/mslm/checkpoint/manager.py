import os, json
import torch
from torch.export.dynamic_shapes import Dim

# Save architecture
# The epoch 1 always has the model.pt as the keys
# The if the checkpoint has different keys it changes from another
# The runs are diferent epochs from the same state_dict

# Nomeclature
# Version    - It is considered if the changes in the architecture are considerable
# Checkpoint - It could be minor changes and if the architure if the same it is executed in with different run file
# Runs       - Same architecture different runs or epochs
# Epoch

class CheckpointManager:
    def __init__(self, base_dir: str, version: int, checkpoint: int):
        self.base = base_dir
        self.v = version

        # Checks if the checkpoint folder is empty or not
        self.ckpt = checkpoint
        # self._compare_architecture_checkpoint(model)

        # Checks if the checkpoint folder is empty or not
        self.run = 1
        # self._check_run_path()
        
    def _path(self, extra=""):
        p = os.path.join(self.base, str(self.v), str(self.ckpt), extra)
        os.makedirs(p, exist_ok=True)
        return p

    def _run_path(self, extra=""):
        p = os.path.join(self.base, str(self.v), str(self.ckpt), str(self.run), extra)
        os.makedirs(p, exist_ok=True)
        return p

    # Checks if there is a run folder
    def _check_run_path(self):
        run = 1
        p = os.path.join(self.base, str(self.v), str(self.ckpt), str(self.run))
        while os.path.isdir(p):
            run = +1
        os.makedirs(p, exist_ok=True)
        return run

    def _compare_architecture_checkpoint(self, model):
        # Returns true or false
        # True if there is a similar architure and the checkpoint
        # False if there is not a similar architecure and the last checkpoint
        self._path()
        p = os.path.join(self.base, str(self.v))
        # Get checkpoints from the versions
        checkpoints_list = os.listdir(p)

        state_dict_model = model.state_dict()
        last_checkpoint = 0

        if len(checkpoints_list) == 0:
            return False

        for checkpoint in checkpoints_list:
            checkpoint_path = os.path.join(p, checkpoint)

            if last_checkpoint < int(checkpoint):
                last_checkpoint = int(checkpoint)

            if os.path.isdir(checkpoint_path) and (
                os.path.isfile(os.path.join(checkpoint_path, "1", "model.pt"))
                or os.path.isfile(os.path.join(checkpoint_path, "1", "1", "model.pt"))
            ):
                model_path = os.path.join(checkpoint_path, "1", "model.pt")

                if os.path.isfile(model_path):
                    model_checkpoint = torch.load(model_path)
                else:
                    model_path = os.path.join(checkpoint_path, "1", "1", "model.pt")
                    model_checkpoint = torch.load(model_path)

                state_dict_model_checkpoint = model_checkpoint.state_dict()

                if state_dict_model.keys() == state_dict_model_checkpoint.keys():
                    self.ckpt = int(checkpoint)
                    return True

                del model_checkpoint

        self.ckpt = int(last_checkpoint) + 1
        return False

    def load_checkpoint(self, model):
        state = torch.load(self._run_path())
        model.load_state_dict(state)
        return model

    def save_checkpoint(self, model, epoch):
        path = self._run_path(str(epoch))
        torch.save(model.state_dict(), os.path.join(path, "checkpoint.pth"))

    def load_model(self, model, path):
        state = torch.load(path)
        model.load_state_dict(state)
        return model

    def save_model(self, model, epoch):
        path = self._path(str(epoch))
        torch.save(model, os.path.join(path, "model.pt"))

    def save_params(self, params):
        p = self._path()
        with open(os.path.join(p, "parameters.json"), "w") as f:
            json.dump(params, f, indent=2)
