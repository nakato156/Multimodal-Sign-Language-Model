import torch._inductor.config
from tqdm import tqdm

from accelerate import Accelerator
import torch
import random

torch.manual_seed(23)
random.seed(23)

from torch import autocast, GradScaler
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter

from src.mslm.utils.early_stopping import EarlyStopping
from src.mslm.checkpoint.manager import CheckpointManager
from src.mslm.training import imitator_loss
import nvtx

torch.autograd.set_detect_anomaly(True)

class Trainer:
    def __init__(self, model, train_loader, val_loader, compile=True, **kwargs):
        #Accelerator module
        self.accelerator = Accelerator()
        self.device = self.accelerator.device

        #Hyperparameters
        self.epochs = kwargs.get("epochs", 100)
        self.learning_rate = kwargs.get("learning_rate", 1e-4)

        #Loggers
        self.log_interval = kwargs.get("log_interval", 5)
        self.writer = SummaryWriter("../outputs/reports/")
        
        #Save and checkpoint
        self.checkpoint_interval = kwargs.get("checkpoint_interval", 5)
        self.ckpt_mgr = CheckpointManager(
            kwargs.get("model_dir", "../outputs/checkpoints"),
            kwargs.get("model_version", 1),
            kwargs.get("checkpoint", 0),
        )

        #Loss Function
        if compile:
            self.criterion = torch.compile(imitator_loss,
                              dynamic=True
            )            
        else:
            self.criterion = imitator_loss

        #Model
        self.model = self.accelerator.prepare_model(model)

        #Dataloaders
        self.train_loader = self.accelerator.prepare_data_loader(train_loader)
        self.val_loader = self.accelerator.prepare_data_loader(val_loader)

        #Stopper
        self.early_stopping = EarlyStopping(patience=100)

        #Optimizer
        self.optimizer = None
        self.scheduler = None

        #Autograd 
        self.dtype_ac = torch.bfloat16 if self.accelerator.mixed_precision == "bf16" else torch.float16

        #Options 
        self.prof = False
        self.distributed = None

    def prepare_optimizer_scheduler(self):
        self.optimizer = self.accelerator.prepare_optimizer(self.optimizer)
        self.scheduler = self.accelerator.prepare_scheduler(self.scheduler)

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
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.epochs,
            eta_min=1e-6,
            last_epoch=-1
        )
        self.prepare_optimizer_scheduler()
        self.prof = prof

        for epoch in tqdm(range(self.epochs), desc="Entrenando", colour="green"):
            train_loss = self._train_epoch(epoch)
            val_loss = self._val(epoch)

            if epoch == 1:
                self.ckpt_mgr.save_model(self.model, epoch)
            elif epoch == self.epochs - 1:
                self.ckpt_mgr.save_model(self.model, epoch)
            elif (epoch % self.checkpoint_interval == 0 and epoch != 0) :
                self.ckpt_mgr.save_model(self.model, epoch)
            elif self.early_stopping.stop:
                self.ckpt_mgr.save_model(self.model, epoch)

            if self.scheduler is not None:
                self.scheduler.step()
            
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
        self.prepare_optimizer_scheduler()

        for epoch in tqdm(range(self.epochs), desc="Entrenando", colour="green"):
            train_loss = self._train_epoch(epoch)
            if rank == 0:
                if epoch == 1:
                    save_model_dist()
                elif (epoch % self.checkpoint_interval == 0 and epoch != 0) or (epoch == self.epochs - 1):
                    save_model_dist()

            val_loss = self.accelerator.gather(torch.tensor(val_loss, device=self.device.type)).mean().item()
            self.scheduler.step(val_loss)
            if self.early_stopping.stop and rank == 0:
                save_model_dist()

            if epoch % self.log_interval == 0:
                tqdm.write(f"\nEpoch: {epoch}.\t Total loss: {train_loss/len(self.train_loader)}")

            #torch.cuda.empty_cache()
        return train_loss, val_loss

    @nvtx.annotate("Train: Train Epoch", color="green")
    def _train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        for keypoints, mask_frames, embeddings, mask_embeddings in self.train_loader:
            torch._dynamo.mark_dynamic(keypoints, 1)
            torch._dynamo.mark_dynamic(mask_frames, 1)
            torch._dynamo.mark_dynamic(embeddings, 1)
            torch._dynamo.mark_dynamic(mask_embeddings, 1)
            
            with self.accelerator.accumulate(self.model):
                loss = self._train_batch(keypoints, mask_frames, embeddings, mask_embeddings)
            if self.distributed is not None:
                loss_tensor = loss.to(self.device)
                self.distributed.all_reduce(loss_tensor, op=self.distributed.ReduceOp.SUM)
                loss = (loss_tensor) / self.distributed.get_world_size()
                if self.distributed.get_rank() == 0:
                    print(f"World-avg train loss: {loss:.4f}")
            else:
                self.writer.add_scalar("Loss/train", loss, epoch)
                total_loss += loss
        if epoch % self.log_interval == 0:
            tqdm.write(f"\nEpoch: {epoch}.\t Total loss: {total_loss.item()/len(self.train_loader)}")

        return total_loss

    def _forward_loss(self, keypoint, mask_frame, embedding, mask_embedding):
        with autocast(device_type=self.device.type, dtype=self.dtype_ac):
            output = self.model(keypoint, mask_frame)
            loss = self.criterion(output, embedding, mask_embedding)
        return loss           

    @nvtx.annotate("Train: Train Batch", color="green")
    def _train_batch(self, keypoint, mask_frame, embedding, mask_embedding):
        if not self.prof:
            loss = self._forward_loss(keypoint, mask_frame, embedding, mask_embedding)
            self.accelerator.backward(loss)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)        
        else:
            with nvtx.annotate("Forward Pass", color="blue"):
                loss = self._forward_loss(keypoint, mask_frame, embedding, mask_embedding)
            with nvtx.annotate("Backward Pass", color="blue"):
                self.accelerator.backward(loss)
            with nvtx.annotate("Update", color="blue"):    
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)        
        return loss

    @nvtx.annotate("Validation Section", color="green")
    def _val(self, epoch):
        self.model.eval()
        val_loss=0
        for keypoints, mask_frames, embeddings, mask_embeddings in self.val_loader:
            loss = self._val_batch(keypoints, mask_frames, embeddings, mask_embeddings)
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
        return final_val_loss

    @nvtx.annotate("Val: Validate Batch", color="green")
    def _val_batch(self, keypoint, mask_frame, embedding, mask_embedding):
        if not self.prof:
            loss = self._forward_loss(keypoint, mask_frame, embedding, mask_embedding)
        else:
            with nvtx.annotate("Val: Forward + Loss", color="blue"):
                loss = self._forward_loss(keypoint, mask_frame, embedding, mask_embedding)
        return loss