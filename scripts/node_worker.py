from src.mslm.distributed import data_pb2, data_pb2_grpc
from src.mslm.utils import create_dataloaders, build_model, run_dt_training

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import os
import grpc
import torch

MASTER_IP = os.environ["MASTER_ADDR"]
MASTER_PORT = os.environ["MASTER_PORT"]

RANK = os.environ["RANK"]
WORLD_SIZE = os.environ["WORLD_SIZE"]

def init_ddp():
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://{MASTER_IP}:{MASTER_PORT}",
        world_size=WORLD_SIZE,
        rank=RANK
    )

    torch.cuda.set_device(RANK % torch.cuda.device_count())
    return dist

def run(hyperparameters, model_parameters):
    channel = grpc.insecure_channel(
        #f"{MASTER_IP}:{MASTER_PORT}",
        "localhost:50051",
        options=[
            ('grpc.max_send_message_length',    100*1024*1024),
            ('grpc.max_receive_message_length', 100*1024*1024),
        ]
    )

    stub = data_pb2_grpc.DataServiceStub(channel)
    hyperparameters = stub.GetHyperparams(data_pb2.Empty())
    model_parameters = stub.GetModelParameters(data_pb2.Empty())

    dist = init_ddp()
    device = torch.device(f"cuda:{RANK % torch.cuda.device_count()}")

    tr_dl, val_dl = create_dataloaders(None, None, None, None, True, f"{MASTER_IP}:{MASTER_PORT}",rank=RANK, world_size=WORLD_SIZE)

    model = build_model(**model_parameters)
    ddp_model = DDP(model, device_ids=[device])    

    run_dt_training(hyperparameters, tr_dl, val_dl, ddp_model, RANK, channel, dist)

if __name__ == "__main__":
    run()