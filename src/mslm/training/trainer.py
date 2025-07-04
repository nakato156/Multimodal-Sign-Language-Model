from tqdm import tqdm
import os

from accelerate import Accelerator
from accelerate.utils import TorchDynamoPlugin
import torch
import random

torch.manual_seed(23)
random.seed(23)

from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import LambdaLR

from src.mslm.utils.early_stopping import EarlyStopping
from src.mslm.checkpoint.manager import CheckpointManager
from src.mslm.training import imitator_loss
import nvtx
from datetime import datetime
import gc

#print("Prueba de configuraciones",
#    torch.backends.cuda.matmul.allow_tf32,
#    torch.backends.cudnn.allow_tf32,
#    torch._dynamo.config.cache_size_limit,
#    torch._dynamo.config.dynamic_shapes,
#    torch._dynamo.config.automatic_dynamic_shapes,
#    
#    os.environ.get('DYNAMIC_SHAPE'),
#    os.environ.get('TORCH_LOGS'),
#    os.environ.get('PYTORCH_CUDA_ALLOC_CONF')
#)

class Trainer:
    def __init__(self, model, train_loader, val_loader, compile=True, save_tb_model=False, **kwargs):
        dynamo_plugin = TorchDynamoPlugin(
            backend="inductor",  # Options: "inductor", "aot_eager", "aot_nvfuser", etc.
            mode="default",      # Options: "default", "reduce-overhead", "max-autotune"
            dynamic=True
        )
        
        #Accelerator module
        self.accelerator = Accelerator(mixed_precision="bf16", dynamo_plugin=dynamo_plugin)
        self.device = self.accelerator.device

        #Hyperparameters
        self.epochs = kwargs.get("epochs", 100)
        self.learning_rate = kwargs.get("learning_rate", 1e-4)

        #Loggers
        self.log_interval = kwargs.get("log_interval", 5)
        self.save_tb_model = save_tb_model
        self.writer = SummaryWriter(f"../outputs/reports/{datetime.now().strftime('%d-%m-%Y-%H-%M-%S')}")
        
        #Save and checkpoint
        self.checkpoint_interval = kwargs.get("checkpoint_interval", 5)
        self.ckpt_mgr = CheckpointManager(
            kwargs.get("model_dir", "../outputs/checkpoints"),
            kwargs.get("model_version", 1),
            kwargs.get("checkpoint", 0),
        )

        #Loss Function
        if compile:
            self.criterion = torch.compile(
                imitator_loss,
                dynamic=True
            )            
        else:
            self.criterion = imitator_loss

        #Model
        self.model = self.accelerator.prepare_model(model)
        self.model = model.to(torch.float32)
        
        #Dataloaders
        self.train_loader = self.accelerator.prepare_data_loader(train_loader)
        self.val_loader = self.accelerator.prepare_data_loader(val_loader)

        #Stopper
        self.early_stopping = EarlyStopping(patience=100)

        #Optimizer
        self.optimizer = None
        self.scheduler = None

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
        
        def linear_warmup_cosine_decay(current_step, warmup_steps, total_steps):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 0.5 * (1.0 + torch.cos(
                torch.tensor((current_step - warmup_steps) / (total_steps - warmup_steps) * 3.1415926535))
            ).item()

        warmup_steps = 5 * len(self.train_loader)  # p.ej. 5 epochs de warm-up
        total_steps = self.epochs * len(self.train_loader)

        lr_lambda = lambda step: linear_warmup_cosine_decay(step, warmup_steps, total_steps)
        self.scheduler = LambdaLR(self.optimizer, lr_lambda=lr_lambda)
        self.prepare_optimizer_scheduler()

        self.prof = prof

        for epoch in tqdm(range(self.epochs), desc="Entrenando", colour="green"):
            train_loss = self._train_epoch(epoch)
            torch.cuda.empty_cache()
            val_loss = self._val(epoch)
            torch.cuda.empty_cache()

            if epoch == 1:
                self.ckpt_mgr.save_model(self.model, epoch)
            elif epoch == self.epochs - 1:
                self.ckpt_mgr.save_model(self.model, epoch)
            elif (epoch % self.checkpoint_interval == 0 and epoch != 0) :
                self.ckpt_mgr.save_model(self.model, epoch)
            elif self.early_stopping.stop:
                self.ckpt_mgr.save_model(self.model, epoch)

            self.scheduler.step()

            if self.early_stopping.stop:
                break

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
            weight_decay=1e-4,
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
        for keypoint, frames_padding_mask, embedding, mask_embedding in self.train_loader:
            if self.save_tb_model and epoch == 1 and not getattr(self, "graph_added", False):
                self.writer.add_graph(self.model, next(iter(self.train_loader))[:2])
                self.graph_added = True           
            
            with self.accelerator.accumulate(self.model):
                self.optimizer.zero_grad(set_to_none=True)        
                loss = self._train_batch(keypoint, frames_padding_mask, embedding, mask_embedding)
            if self.distributed is not None:
                loss_tensor = loss.to(self.device)
                self.distributed.all_reduce(loss_tensor, op=self.distributed.ReduceOp.SUM)
                loss = (loss_tensor) / self.distributed.get_world_size()
                if self.distributed.get_rank() == 0:
                    print(f"World-avg train loss: {loss:.4f}")
            else:
                total_loss += loss.detach()
                
        final_train_loss = total_loss.item()/len(self.train_loader)
        self.writer.add_scalar("Loss/train", final_train_loss, epoch)
        if epoch % self.log_interval == 0:
            tqdm.write(f"\nEpoch: {epoch}.\t Total loss: {final_train_loss}")

        gc.collect()
        return total_loss

    def _forward_loss(self, keypoint, frames_padding_mask, embedding, mask_embedding):
        with self.accelerator.autocast():
            keypoint = keypoint.to(torch.float32)
            embedding = embedding.to(torch.float32)

            output = self.model(keypoint, frames_padding_mask)
            loss = self.criterion(output, embedding, mask_embedding)
            
        del keypoint, frames_padding_mask, embedding, mask_embedding, output
        return loss

    @nvtx.annotate("Train: Train Batch", color="green")
    def _train_batch(self, keypoint, frames_padding_mask, embedding, mask_embedding):
        if not self.prof:
            try:
                with torch.autograd.set_detect_anomaly(True):
                    loss = self._forward_loss(keypoint, frames_padding_mask, embedding, mask_embedding)
                    self.accelerator.backward(loss)
            except Exception as e:
                print(e)
                print("Keypoint: ", keypoint.shape, "\n", 
                      "MaskFrame: ", frames_padding_mask.shape, "\n",
                      "Embedding: ", embedding.shape, "\n",
                      "Mask Embedding: ", mask_embedding.shape
                      )
            self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
        else:
            with torch.autograd.set_detect_anomaly(True):
                with nvtx.annotate("Forward Pass", color="blue"):
                    loss = self._forward_loss(keypoint, frames_padding_mask, embedding, mask_embedding)
                with nvtx.annotate("Backward Pass", color="blue"):
                        self.accelerator.backward(loss)
            with nvtx.annotate("Update", color="blue"):    
                self.accelerator.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
        return loss.detach()

    @nvtx.annotate("Validation Section", color="green")
    def _val(self, epoch):
        self.model.eval()
        val_loss=0
        for keypoint, frames_padding_mask, embedding, mask_embedding in self.val_loader:        
            loss = self._val_batch(keypoint, frames_padding_mask, embedding, mask_embedding)
            if self.distributed is not None:
                loss_tensor = loss.to(self.device)
                self.distributed.all_reduce(loss_tensor, op=self.distributed.ReduceOp.SUM)
                val_loss = (loss_tensor) / self.distributed.get_world_size()
                if self.distributed.get_rank() == 0:
                    print(f"World-avg val loss: {loss:.4f}")
            else:
                val_loss += loss.detach()
        final_val_loss = val_loss.item() / len(self.val_loader)
        self.writer.add_scalar("Loss/val", final_val_loss, epoch)

        if epoch % self.log_interval == 0:
            tqdm.write(f"Validation loss: {final_val_loss}")

        self.early_stopping(final_val_loss)
        return final_val_loss

    @nvtx.annotate("Val: Validate Batch", color="green")
    def _val_batch(self, keypoint, frames_padding_mask, embedding, mask_embedding):
        if not self.prof:
            loss = self._forward_loss(keypoint, frames_padding_mask, embedding, mask_embedding)
        else:
            with nvtx.annotate("Val: Forward + Loss", color="blue"):
                loss = self._forward_loss(keypoint, frames_padding_mask, embedding, mask_embedding)
        return loss