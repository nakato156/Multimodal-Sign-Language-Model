from src.mslm.distributed import data_pb2, data_pb2_grpc
from src.mslm.utils.setup_train import setup_paths
from src.mslm.checkpoint.manager import CheckpointManager
from src.mslm.dataloader import collate_fn
from src.mslm.utils import prepare_datasets, ConfigLoader

import io
import grpc
import torch
from concurrent import futures
from torch.utils.data import DataLoader, DistributedSampler

BATCH_SIZE = 8
NUM_WORKERS = 4

class DataServiceServicer(data_pb2_grpc.DataServiceServicer):
    def __init__(self, 
                epochs: int,
                batch_size: int,
                checkpoint_interval: int,
                log_interval: int,
                train_ratio: float = 0.8):
        
        _, _, h5_file = setup_paths()
        device = "cuda"

        self.model_parameters = ConfigLoader("config/model/config.toml").load_config()
        self.model_parameters.update({
            "device": device if self.model_parameters.get("device") == "auto" else self.model_parameters.get("device", device),
            "input_size": 250 * 2,
            "output_size": 3072,
        })

        self.train_config = ConfigLoader("config/training/train_config.toml").load_config()
        train_ratio = self.train_config.get("train_ratio", train_ratio)
        self.train_config.update({
            "learning_rate": self.train_config.get("learning_rate", 0.00238),
            "epochs": epochs if epochs else self.train_config.get("epochs", 100),
            "batch_size": batch_size if batch_size else self.train_config.get("batch_size", 32),
            "checkpoint_interval": checkpoint_interval if checkpoint_interval else self.train_config.get("checkpoint_interval", 5),
            "log_interval": log_interval if log_interval else self.train_config.get("log_interval", 2),
            "train_ratio": train_ratio,
            "validation_ratio": round(1 - train_ratio, 2),
            "device": device if self.model_parameters.get("device") == "auto" else self.model_parameters.get("device", device),
        })

        self.tr_ds, self.val_ds = prepare_datasets(h5_file, train_ratio)
        self.ckpt_mgr = CheckpointManager(
                "../outputs/checkpoints",
                self.train_config["model_version"],
                self.train_config["checkpoint"],
        )

    def StreamData(self, request, context):
        if request.split =="train":
            sampler = DistributedSampler(
                self.tr_ds, 
                num_replicas=request.world_size,
                rank=request.rank,
                shuffle=True
            )
            loader = DataLoader(
                self.tr_ds,
                batch_size=BATCH_SIZE,
                sampler=sampler,
                num_workers=NUM_WORKERS,
                collate_fn=collate_fn
            )

            for batch in loader:
                keypoint_tensor = batch[0]
                keypoint_mask_tensor = batch[1]
                embedding_tensor = batch[2]
                embedding_mask_tensor = batch[3]
                yield data_pb2.StreamResponse(
                    keypoint=data_pb2.Tensor(
                        bytes = keypoint_tensor.numpy().tobytes(),
                        shape = list(keypoint_tensor.shape),
                        dtype = str(keypoint_tensor.dtype)
                    ),
                    keypoint_mask=data_pb2.Tensor(
                        bytes = keypoint_mask_tensor.numpy().tobytes(),
                        shape = list(keypoint_mask_tensor.shape),
                        dtype = str(keypoint_mask_tensor.dtype)
                    ),
                    embedding=data_pb2.Tensor(
                        bytes = embedding_tensor.numpy().tobytes(),
                        shape = list(embedding_tensor.shape),
                        dtype = str(embedding_tensor.dtype)
                    ),
                    embedding_mask=data_pb2.Tensor(
                        bytes = embedding_mask_tensor.numpy().tobytes(),
                        shape = list(embedding_mask_tensor.shape),
                        dtype = str(embedding_mask_tensor.dtype)
                    ),
                )
            
    def SaveModel(self, request, context):
        buf = io.BytesIO(request.model_bytes)
        model = torch.load(buf, map_location="cpu")
        self.ckpt_mgr.save_model(model, epoch=request.model_name)
        return data_pb2.SaveModelResponse(success=True, 
                                          message=f"Model saves as {request.model_name}")

    def GetHyperparams(self, request, context):
        return data_pb2.Hyperparams(
            learning_rate=self.train_config["learning_rate"],
            model_version=self.train_config["model_version"],
            epochs=self.train_config["epochs"]
        )

    def GetModelParameters(self, request, context):
        return data_pb2.ModelParameters(
            output_size=self.model_parameters["output_size"],
            hidden_size=self.model_parameters["hidden_size"],
            nhead=self.model_parameters["nhead"],
            ff_dim=self.model_parameters["ff_dim"],
            n_layers=self.model_parameters["n_layers"],
        )

def serve(args, host="0.0.0.0", port=50051):
    grpc_opts = [
        ('grpc.max_send_message_length', 100 * 1024 * 1024),
        ('grpc.max_receive_message_length', 100 * 1024 * 1024),
    ]

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=grpc_opts
        )
    data_pb2_grpc.add_DataServiceServicer_to_server(DataServiceServicer(args.epochs, args.batch_size, args.checkpoint_interval, args.log_interval), server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    print("The server has started")
    server.wait_for_termination()

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train a model.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs to train.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--checkpoint_interval", type=int, default=5, help="Interval for saving checkpoints.")
    parser.add_argument("--log_interval", type=int, default=2, help="Interval for logging training progress.")
    args = parser.parse_args()
    
    serve(args)