from src.mslm.distributed import data_pb2, data_pb2_grpc
from src.mslm.utils import create_dataloaders, build_model, run_dt_training

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import os
import grpc
import torch
from dataclasses import asdict

DDP_IP = os.environ["DDP_ADDR"]
DDP_PORT = os.environ["DDP_PORT"]

GRPC_IP = os.environ["GRPC_ADDR"]
GRPC_PORT = os.environ["GRPC_PORT"]

RANK = int(os.environ["RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])

print(DDP_IP, DDP_PORT)
print(GRPC_IP, GRPC_PORT)
print(RANK, WORLD_SIZE)

def init_ddp():
    torch.cuda.set_device(0)

    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{DDP_IP}:{DDP_PORT}",
        world_size=WORLD_SIZE,
        rank=RANK
    )

    assert dist.is_initialized(), "DDP init failed!"

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print(f"[Rank {rank}/{world_size}] GPU {torch.cuda.current_device()} allocated.")

    # Barrier so everyone prints before proceeding
    dist.barrier(device_ids=[torch.cuda.current_device()])
    if rank == 0:
        print(">>> All ranks have initialized. DDP is working!")

    return dist

def run():
    channel = grpc.insecure_channel(
        f"{GRPC_IP}:{GRPC_PORT}",
        #"localhost:50051",
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

    hp_kwargs = {
    field.name: getattr(hyperparameters, field.name)
    for field in hyperparameters.DESCRIPTOR.fields
    }

    mp_kwargs = {
    field.name: getattr(model_parameters, field.name)
    for field in model_parameters.DESCRIPTOR.fields
    }

    mp_kwargs["device"] = device
    mp_kwargs["input_size"] = 250*2

    print(hyperparameters)
    print(model_parameters)

    tr_dl, val_dl = create_dataloaders(None, None, hp_kwargs["batch_size"], None, True, f"{GRPC_IP}:{GRPC_PORT}",rank=RANK, world_size=WORLD_SIZE)

    model = build_model(**mp_kwargs)
    ddp_model = DDP(model, device_ids=[device])    

    run_dt_training(hp_kwargs, tr_dl, val_dl, ddp_model, RANK, channel, dist, stub)

if __name__ == "__main__":
    run()