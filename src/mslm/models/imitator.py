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
        nhead: int = 8,
        ff_dim: int = 3136,
        n_layers: int = 2,
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
        


        # --- Bloque de entrada ---
        self.giorgio = nn.AdaptiveAvgPool1d(output_size=128)  # Giorgio es un pooler que maneja los frames variables de entrada

        self.linear_feat = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2)
        )

        self.linear_seq = nn.Sequential(
            nn.Conv1d(hidden_size//2, 128, kernel_size=3, padding=1),
            nn.LayerNorm(pool_dim),
            nn.GELU(),
            nn.Linear(pool_dim, pool_dim),
            nn.LayerNorm(pool_dim)
        )
    
        # Volvemos a hidden_size
        self.linear_hidden = nn.Linear(pool_dim, hidden_size)
        self.norm4         = nn.LayerNorm(hidden_size)

        # Positional Encoding + Transformer
        self.pe          = PositionalEncoding(hidden_size)
        encoder_layer    = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=ff_dim,
            batch_first=True,
            dropout=0.1,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Proyección final por paso de tiempo
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        B, T, D, C = x.shape
        # x -> [batch_size, T, input_size]
        x = x.view(B, T,  D * C)            # [B, T, input_size]
        
        x = self.linear_feat(x)             # [B, T, hidden//2]

        x = x.transpose(1, 2)               # [B, hidden//2, T]
        x = self.giorgio(x)                 # Giorgio -> [B, hidden//2, 525]
        x  = self.linear_seq(x)             # [B, hidden//2, pool_dim]
        x = x.transpose(1, 2)               # [B, pool_dim, hidden//2]

        x = self.linear_hidden(x)           # [B, pool_dim, hidden]
        x = self.norm4(x)                   # [B, pool_dim, hidden]
        x = F.relu(x)                       # [B, pool_dim, hidden]

        x = self.pe(x)
        x = self.transformer(x)             # [B, pool_dim, hidden]

        x = self.proj(x)                    # [B, pool_dim, output_size]
        return x