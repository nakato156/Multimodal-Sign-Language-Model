import torch
import torch.nn as nn
from .components.positional_encoding import PositionalEncoding
import torch.nn.functional as F

class Imitator(nn.Module):
    def __init__(
        self,
        input_size: int = 1086,
        hidden_size: int = 512,
        T_size: int = 525,
        output_size: int = 3072,
        nhead: int = 32,
        ff_dim: int = 1024,
        n_layers: int = 8,
        pool_dim: int = 128,
    ):
        super().__init__()

        self.cfg = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "T_size": T_size,
            "output_size": output_size,
            "nhead": nhead,
            "ff_dim": ff_dim,
            "n_layers": n_layers,
            "pool_dim": pool_dim
        }
        
        self.linear = nn.Linear(input_size, hidden_size)
        nn.init.xavier_uniform_(self.linear.weight)
        self.norm = nn.LayerNorm(hidden_size)

        self.pe = PositionalEncoding(d_model=hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, dim_feedforward=ff_dim, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.temporal_adjuster = nn.Sequential(
            nn.Linear(T_size, pool_dim),
            nn.ReLU(),
            nn.Linear(pool_dim, pool_dim),
        )

        self.linear_out = nn.Linear(hidden_size, output_size)
        nn.init.xavier_uniform_(self.linear_out.weight)
        

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        # x -> [batch_size, T, input_size]
        B, T, D, C = x.shape
        x = x.view(B, T,  D * C)
        x = F.gelu(self.linear(x))
        x = self.norm(x)

        x += self.pe(x)
        x = self.transformer(x)

        x = x.transpose(1, 2)           # [B, hidden, 525]
        x = self.temporal_adjuster(x)   # [B, hidden, 128]
        x = x.transpose(1, 2)
        
        x = self.linear_out(x)
        return x