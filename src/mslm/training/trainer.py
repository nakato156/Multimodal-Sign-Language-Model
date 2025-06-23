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

class Trainer:
    def __init__(self, model, train_loader, val_loader, **kwargs):
        self.LOG = kwargs.get("log", False)
        
        self.device = kwargs.get("device", "cuda")
        print(self.device)
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
        self.prof = False
        self.distributed = None
        self.dtype_ac = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else None

        self.optimizer = None
        self.scheduler = None

        self._train_batch = torch.compile(
            self._train_batch,
            backend="aot_eager",
            dynamic=True
        )

        self._val_batch = torch.compile(
            self._val_batch,
            backend="aot_eager",
            dynamic=True
        )

    @nvtx.annotate("Training Section", color="green")
    def train(self, prof = False):

        """Entrena el modelo Imitator.
        returns:
            train_loss: float, loss de entrenamiento
            val_loss: float, loss de validación
        """
        print("LR:", self.learning_rate)
        self.optimizer = AdamW(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=1e-3,
            foreach=True
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            min_lr=1e-7
        )
        self.prof = prof
        train_loss = 0
        val_loss = 0

        for epoch in tqdm(range(self.epochs), desc="Entrenando", colour="green"):
            train_loss = self._train_epoch(epoch)
            if epoch == 1:
                self.ckpt_mgr.save_model(self.model, epoch)
            elif (epoch % self.checkpoint_interval == 0 and epoch != 0) or (epoch == self.epochs - 1):
                self.ckpt_mgr.save_model(self.model, epoch)

            val_loss = self._val(epoch)
            self.scheduler.step(val_loss)
            if self.early_stopping.stop:
                self.ckpt_mgr.save_model(self.model, epoch)

            torch.cuda.empty_cache()
        return train_loss, val_loss

    @nvtx.annotate("Distributed Training Section", color="green")
    def train_dist(self, rank, dist, stub):
        from src.mslm.distributed import data_pb2, data_pb2_grpc
        import io

        """Entrena el modelo Imitator distribuido.
        returns:
            train_loss: float, loss de entrenamiento
            val_loss: float, loss de validación
        """
        self.distributed = dist
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
        self.optimizer = AdamW(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=1e-3,
            foreach=True
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=2,
            min_lr=1e-7
        )

        for epoch in tqdm(range(self.epochs), desc="Entrenando", colour="green"):
            train_loss = self._train_epoch(epoch)
            if rank == 0:
                if epoch == 1:
                    save_model_dist()
                elif (epoch % self.checkpoint_interval == 0 and epoch != 0) or (epoch == self.epochs - 1):
                    save_model_dist()

            val_loss = self._val(epoch)
            self.scheduler.step(val_loss)
            if self.early_stopping.stop and rank == 0:
                save_model_dist()

            if epoch % self.log_interval == 0:
                tqdm.write(f"\nEpoch: {epoch}.\t Total loss: {train_loss.item()/len(self.train_loader)}")

            torch.cuda.empty_cache()
        return train_loss, val_loss

    @nvtx.annotate("Train: Train Epoch", color="green")
    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0

        for keypoints, mask_frames, embeddings, mask_embeddings in self.train_loader:
            keypoint = keypoints.to(self.device, non_blocking=True)
            embedding = embeddings.to(self.device, non_blocking=True)
            mask_frame = mask_frames.to(self.device, non_blocking=True)
            mask_embedding = mask_embeddings.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)        
            loss = self._train_batch(keypoint, mask_frame, embedding, mask_embedding)
            if self.distributed is not None:
                loss_tensor = loss.to(self.device)
                self.distributed.all_reduce(loss_tensor, op=self.distributed.ReduceOp.SUM)
                loss = (loss_tensor) / self.distributed.get_world_size()
                if self.distributed.get_rank() == 0:
                    print(f"World-avg train loss: {loss:.4f}")
            else:
                self.writer.add_scalar("Loss/train", loss, epoch)
                total_loss += loss.detach()
        if epoch % self.log_interval == 0:
            tqdm.write(f"\nEpoch: {epoch}.\t Total loss: {total_loss.item()/len(self.train_loader)}")

        return total_loss

    @nvtx.annotate("Train: Train Batch", color="green")
    def _train_batch(self, keypoint, mask_frame, embedding, mask_embedding):
        if not self.prof:        
            with autocast(device_type=self.device, dtype=self.dtype_ac):
                output = self.model(keypoint, mask_frame)
                loss = self.criterion(output, embedding, mask_embedding)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            self.optimizer.step()
        else:        
            with nvtx.annotate("Forward Pass", color="blue"):
                with autocast(device_type=self.device, dtype=self.dtype_ac):
                    output = self.model(keypoint, mask_frame)
                    loss = self.criterion(output, embedding, mask_embedding)
            with nvtx.annotate("Backward Pass", color="blue"):
                self.scaler.scale(loss).backward()
            with nvtx.annotate("Update", color="blue"):    
                self.scaler.unscale_(self.optimizer)
                self.optimizer.step()

        loss_detach = loss.detach()
        self.scaler.update()
        return loss_detach

    @nvtx.annotate("Validation Section", color="green")
    def _val(self, epoch):
        self.model.eval()
        val_loss=0
        with torch.inference_mode():
            for keypoints, mask_frames, embeddings, mask_embeddings in self.val_loader:
                keypoint = keypoints.to(self.device, non_blocking=True)
                embedding = embeddings.to(self.device, non_blocking=True)
                mask_frame = mask_frames.to(self.device, non_blocking=True)
                mask_embedding = mask_embeddings.to(self.device, non_blocking=True)

                loss = self._val_batch(keypoint, mask_frame, embedding, mask_embedding)
                if self.distributed is not None:
                    loss_tensor = loss.to(self.device)
                    self.distributed.all_reduce(loss_tensor, op=self.distributed.ReduceOp.SUM)
                    val_loss = (loss_tensor) / self.distributed.get_world_size()
                    if self.distributed.get_rank() == 0:
                        print(f"World-avg val loss: {loss:.4f}")

                else:
                    val_loss += loss.detach()
            final_val_loss = val_loss.item() / len(self.val_loader)

            if epoch % self.log_interval == 0:
                tqdm.write(f"Validation loss: {final_val_loss}")

        self.early_stopping(final_val_loss)
        torch.cuda.empty_cache()
        return final_val_loss

    @nvtx.annotate("Val: Validate Batch", color="green")
    def _val_batch(self, keypoint, mask_frame, embedding, mask_embedding):
        if not self.prof:
            with autocast(device_type=self.device, dtype=self.dtype_ac):
                output = self.model(keypoint, mask_frame)
                loss = self.criterion(output.to(dtype=self.dtype_ac), embedding, mask_embedding)
        else:
            with nvtx.annotate("Val: Forward + Loss", color="blue"):
                with autocast(device_type=self.device, dtype=self.dtype_ac):
                    output = self.model(keypoint, mask_frame)
                    loss = self.criterion(output.to(dtype=self.dtype_ac), embedding, mask_embedding)
        
        return loss.detach()