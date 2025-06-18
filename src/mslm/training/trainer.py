import torch._inductor.config
from tqdm import tqdm

import torch
from torch import autocast, GradScaler
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from src.mslm.utils.early_stopping import EarlyStopping
from src.mslm.checkpoint.manager import CheckpointManager
from src.mslm.training import imitator_loss
import nvtx

torch._inductor.config.triton.cudagraph_skip_dynamic_graphs = True

class Trainer:
    def __init__(self, model, train_loader, val_loader, **kwargs):
        self.LOG = kwargs.get("log", False)
        
        self.device = kwargs.get("device", "cuda")
        self.epochs = kwargs.get("epochs", 100)
        self.learning_rate = kwargs.get("learning_rate", 1e-4)
        self.log_interval = kwargs.get("log_interval", 5)
        self.checkpoint_interval = kwargs.get("checkpoint_interval", 5)
        self.model = model.to(self.device)
        self.ckpt_mgr = CheckpointManager(
            kwargs.get("model_dir", "../outputs/checkpoints"),
            kwargs.get("model_version", 1),
            kwargs.get("checkpoint", 0),
        )
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.writer = SummaryWriter("../outputs/reports/")
        self.scaler = GradScaler(device=self.device)
        self.early_stopping = EarlyStopping(patience=100)

        self.criterion = imitator_loss

    @nvtx.annotate("Training Section", color="green")
    def train(self):
        """Entrena el modelo Imitator.
        returns:
            train_loss: float, loss de entrenamiento
            val_loss: float, loss de validación
        """
        print("LR:", self.learning_rate)
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            min_lr=1e-7
        )

        train_loss = 0
        val_loss = 0

        for epoch in tqdm(range(self.epochs), desc="Entrenando", colour="green"):
            train_loss = self._train_epoch(epoch, optimizer, scheduler=None)
            if epoch == 1: 
                self.ckpt_mgr.save_model(self.model, 1)
            elif (epoch % self.checkpoint_interval == 0 and epoch != 0) or (epoch == self.epochs - 1):
                self.ckpt_mgr.save_model(self.model, epoch)

            val_loss = self._validate(epoch)
            scheduler.step(val_loss)
            if self.early_stopping.stop:
                self.ckpt_mgr.save_model(self.model, epoch)
                break
        return train_loss, val_loss

    @nvtx.annotate("Training Section", color="green")
    def train_dist(self, rank, channel, dist):
        """Entrena el modelo Imitator.
        returns:
            train_loss: float, loss de entrenamiento
            val_loss: float, loss de validación
        """

        from src.mslm.distributed import data_pb2, data_pb2_grpc
        import io
        
        stub = data_pb2_grpc.DataServiceServicer(channel)

        def save_model_dist():
                buf = io.BytesIO()
                torch.save(self.model, buf)
                req = data_pb2.SaveModelRequest(
                    model_bytes=buf.getvalue(),
                    model_name = f"{epoch}"
                )
                resp = stub.SaveModel(req)
                print("=== SAVE MODEL ===")
                print(" success:", resp.success)
                print(" message:", resp.message)

        print("LR:", self.learning_rate)
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            min_lr=1e-7
        )

        train_loss = 0
        val_loss = 0

        for epoch in tqdm(range(self.epochs), desc="Entrenando", colour="green"):
            train_loss = self._train_epoch(epoch, optimizer, scheduler=None, distributed=True, dist=dist)
            if rank == 0:
                if epoch == 1: 
                    save_model_dist()
                elif (epoch % self.checkpoint_interval == 0 and epoch != 0) or (epoch == self.epochs - 1):
                    save_model_dist()

            val_loss = self._validate(epoch, True, dist)

            scheduler.step(val_loss)
            if self.early_stopping.stop and rank == 0:
                save_model_dist()
                break
        return train_loss, val_loss
    
    def _train_epoch(self, epoch, optimizer, scheduler=None, distributed=False, dist=None):
        self.model.train()
        total_loss = 0

        for data, mask_frames, embeddings, mask_embeddings in self.train_loader:
            optimizer.zero_grad(set_to_none=True)

            with nvtx.annotate("Data to CUDA", color="yellow"):
                data = data.to(self.device)
                embeddings = embeddings.to(self.device)
                mask_frames = mask_frames.to(self.device)
                mask_embeddings = mask_embeddings.to(self.device)

                data = data.contiguous()
                mask_frames = mask_frames.contiguous()

            with nvtx.annotate("Forward Pass", color="blue"):
                #Change to bfloat16 if the GPU used is with Ampere Architecture or Higher
                dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else None

                with autocast(device_type=self.device, dtype=dtype):
                    output = self.model(data, mask_frames)
                    # print(mask_embeddings.shape, embeddings.shape, output.shape)

                    loss = self.criterion(output, embeddings, mask_embeddings)

            with nvtx.annotate("Backward Pass", color="blue"):
                total_loss += loss.detach()
                final_loss = total_loss.item()

            self.writer.add_scalar("Loss/train", loss, epoch)

            if distributed:
                loss_tensor = torch.tensor(loss.item(), device=self.device)
                dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                loss = loss_tensor.item() / dist.get_world_size()

                if dist.get_rank() == 0:
                    print(f"World-avg loss: {loss:.4f}")

            with nvtx.annotate("Update", color="blue"):
                self.scaler.scale(loss).backward()
                
                self.scaler.step(optimizer)
                self.scaler.update()
                if scheduler:
                    scheduler.step()

        torch.cuda.empty_cache()

        if epoch % self.log_interval == 0:
            tqdm.write(f"\nEpoch: {epoch}.\t Total loss: {final_loss/len(self.train_loader)}")

        return final_loss

    @nvtx.annotate("Validation Section", color="blue")
    def _validate(self, epoch, distributed=False, dist=None):
        with nvtx.annotate("Prueba de Validacion", color="blue"):
            with torch.no_grad():
                self.model.eval()
                val_loss = 0
                
                for data, mask_frames, embeddings, mask_embeddings in self.val_loader:
                    data = data.to(self.device)
                    embeddings = embeddings.to(self.device)
                    mask_frames = mask_frames.to(self.device)
                    mask_embeddings = mask_embeddings.to(self.device)
                                        
                    data = data.contiguous()
                    mask_frames = mask_frames.contiguous()

                    output = self.model(data, mask_frames)
                    loss = self.criterion(output.to(dtype=torch.bfloat16), embeddings, mask_embeddings)
                    val_loss += loss.detach()

                    # del output, data, embeddings, cos_sim
                torch.cuda.empty_cache()
                    
                final_val_loss = val_loss.item() / len(self.val_loader)
                if epoch % self.log_interval == 0:
                    tqdm.write(f"Validation Loss: {final_val_loss}" )

                if distributed:
                    loss_tensor = torch.tensor(final_val_loss.item(), device=self.device)
                    dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
                    final_val_loss = loss_tensor.item() / dist.get_world_size()

                    if dist.get_rank() == 0:
                        print(f"World-avg loss: {loss:.4f}")

                self.early_stopping(final_val_loss)
                
                return final_val_loss