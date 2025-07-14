import torch
import torch.nn as nn
from torchvision.ops import stochastic_depth
from torch.utils.checkpoint import checkpoint

from .components.positional_encoding import PositionalEncoding

#from apex.normalization import FusedLayerNorm
#from apex.contrib.multihead_attn import FlashMultiheadAttention
#
#from xformers.ops import memory_efficient_attention as FlashAttention

class Imitator(nn.Module):
    def __init__(
        self,
        input_size: int = 133*2,
        hidden_size: int = 512,
        output_size: int = 3072,
        nhead: int = 8,
        ff_dim: int = 1024,
        n_layers: int = 2,
        max_seq_length: int = 301,
    ):
        super().__init__()

        self.cfg = {
            "input_size": input_size,
            "hidden_size": hidden_size,
            "output_size": output_size,
            "nhead": nhead,
            "ff_dim": ff_dim,
            "n_layers": n_layers,
            "max_seq_length": max_seq_length
        }

        # --- Bloque de entrada ---

        self.linear_feat = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_size // 2)
        )

        pool_dim = 256
        # linear sequencer
        self.conv1  = nn.Conv1d(hidden_size//2, pool_dim, kernel_size=3, padding=1)
        self.ln1    = nn.LayerNorm(pool_dim)
        self.act1   = nn.GELU()
        self.conv2  = nn.Conv1d(pool_dim, pool_dim, kernel_size=1)
        self.ln2    = nn.LayerNorm(pool_dim)
        self.act2   = nn.GELU()
    
        # Volvemos a hidden_size
        self.linear_hidden = nn.Linear(pool_dim, hidden_size)

        # Positional Encoding + Transformer
        self.pe          = PositionalEncoding(hidden_size, dropout=0.2)
        encoder_layer    = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=ff_dim,
            batch_first=True,
            dropout=0.4,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Proyección final por paso de tiempo
        self.proj = nn.Linear(hidden_size, output_size)

        self.token_queries = nn.Parameter(torch.randn(max_seq_length, output_size))  # [1, output_size]
        # Queries = E_tokens [n_tokens × B × d], Keys/Values = frames_repr [T' × B × d]
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=output_size,
            num_heads=nhead,
            dropout=0.1,
            batch_first=True,
        )

        #self.cross_attn = FlashAttention(
        #    embed_dim=output_size,
        #    num_heads=nhead,
        #    dropout=0.1,
        #    batch_first=True
        #)

        self.norm_attn = nn.LayerNorm(output_size)

        self.proj_final = nn.Sequential(
            nn.Linear(output_size, output_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(output_size * 2, output_size)
        )

    def transformer_checkpoint(self, x, mask):
        return self.transformer(x, src_key_padding_mask=mask)

    def forward(self, x:torch.Tensor, frames_padding_mask:torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of frames
        returns: Tensor of embeddings for each token (128 tokens of frames)
        """

        B, T, D, K = x.shape                # x -> [batch_size, T, input_size]
        x = x.view(B, T,  D * K)            # [B, T, input_size]
        
        x = self.linear_feat(x)             # [B, T, hidden//2]

        x = x.transpose(1, 2)               # [B, hidden//2, T]
        # se mantiene T' = T o reducirdo a pool_dim
        x  = self.conv1(x)                  # [B, hidden//2, pool_dim]
        x = x.transpose(1, 2)               # [B, pool_dim, hidden//2]
        x = self.ln1(x)                     # [B, pool_dim, hidden//2]
        x = self.act1(x)                    # [B, pool_dim, hidden//2]
        x = x.transpose(1, 2)               # [B, hidden//2, pool_dim]

        x = self.conv2(x)                   # [B, hidden//2, pool_dim]
        x = x.transpose(1, 2)               # [B, pool_dim, hidden//2]
        x = self.ln2(x)                     # [B, pool_dim, hidden//2]
        x = self.act2(x)                    # [B, pool_dim, hidden//2]

        x = self.linear_hidden(x)           # [B, pool_dim, hidden]

        x = self.pe(x)
        if self.training:
            x = checkpoint(self.transformer_checkpoint, x, frames_padding_mask, use_reentrant=False)
        else:
            x = self.transformer_checkpoint(x, frames_padding_mask)  # [B, pool_dim, hidden]

        M = self.proj(x)     # [B, pool_dim, output_size]
        # M = M.masked_fill(frames_padding_mask.unsqueeze(-1), 0.0)
        
        Q = self.token_queries.unsqueeze(0).expand(B, -1, -1)   # [B, n_tokens, output_size]
    
        attn_out, attn_w = self.cross_attn(
            query=Q,
            key=M,
            value=M,
            key_padding_mask=frames_padding_mask
        )  # [B, n_tokens, output_size]
        x = self.norm_attn(Q + attn_out)
        # print(f"Attention output shape: {attn_out.shape}, Q shape: {Q.shape}, M shape: {M.shape}")
        #x = x + stochastic_depth(self.proj_final(attn_out), p=0.2, mode="row")        # [B, n_tokens, output_size]
        return x