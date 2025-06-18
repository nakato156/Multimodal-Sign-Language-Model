import torch
from torch.utils.data import IterableDataset
import grpc
from src.mslm.distributed import data_pb2, data_pb2_grpc

class GRPCDataset(IterableDataset):
    def __init__(self, address, rank, world_size, split):
        channel = grpc.insecure_channel(address)
        self.stub = data_pb2_grpc.DataServiceStub(channel)
        self.request = data_pb2.StreamRequest(rank=rank, world_size=world_size, split=split)

    def __iter__(self):
        for response in self.stub.StreamData(self.request):
            keypoint = torch.frombuffer(response.keypoint.bytes, dtype=torch.float32)
            keypoint = keypoint.reshape(response.keypoint.shape)

            embedding = torch.frombuffer(response.embedding.bytes, dtype=torch.float32)
            embedding = embedding.reshape(response.embedding.shape)

            yield keypoint, embedding