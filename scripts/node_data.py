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
    def __init__(self, train_ratio = 0.8, **kwargs):
        _, _, h5_file = setup_paths()
        train_config = ConfigLoader("config/training/train_config.toml").load_config()
        train_ratio = train_config.get("train_ratio", train_ratio)
        self.tr_ds, self.val_ds = prepare_datasets(h5_file, train_ratio)
        self.ckpt_mgr = CheckpointManager(
                kwargs.get("model_dir", "../outputs/checkpoints"),
                kwargs.get("model_version", 1),
                kwargs.get("checkpoint", 0),
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

def serve(host="0.0.0.0", port=50051):
    grpc_opts = [
        ('grpc.max_send_message_length', 100 * 1024 * 1024),
        ('grpc.max_receive_message_length', 100 * 1024 * 1024),
    ]

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=grpc_opts
        )
    data_pb2_grpc.add_DataServiceServicer_to_server(DataServiceServicer(), server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    print("The server has started")
    server.wait_for_termination()

if __name__ == "__main__":
    serve()