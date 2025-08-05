import torch
import torch.nn as nn
from .components import TransformerEncoderLayerRoPE
from .components.stgcn import STGCNBlock, partition_adjacency
from torch.utils.checkpoint import checkpoint

class Imitator(nn.Module):
    def __init__(
        self,
        A,  # raw adjacency matrix, no partitioning
        input_size: int,
        hidden_size: int = 512,
        output_size: int = 3072,
        nhead: int = 8,
        ff_dim: int = 1024,
        n_layers: int = 2,
        max_seq_length: int = 20, # cambiar
        encoder_dropout: int = 0.4,
        multihead_dropout: int = 0.1,
        pool_dim: int = 256,
    ):
        super().__init__()

        self.cfg = {
            "A": A,
            "input_size": input_size,
            "output_size": output_size,
            "hidden_size": hidden_size,
            "nhead": nhead,
            "ff_dim": ff_dim,
            "n_layers": n_layers,
            "max_seq_length": max_seq_length,
            "pool_dim": pool_dim,
            "encoder_dropout": encoder_dropout,
            "multihead_dropout": multihead_dropout,
        }

        print("Model Parameters: ", self.cfg)
    
        # --- Bloque de entrada ---
        A = partition_adjacency(A)
        self.stgcn = STGCNBlock(2, hidden_size // 2, A, kernel_size=3, stride=1)
    
        # Volvemos a hidden_size
        self.linear_hidden = nn.Sequential(
            nn.Conv2d(3 * (hidden_size // 2), hidden_size, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(hidden_size)
        )

        # Positional Encoding + Transformer
        encoder_layer    = TransformerEncoderLayerRoPE(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=encoder_dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.token_queries = nn.Parameter(torch.randn(max_seq_length, hidden_size))  # [1, hidden_size]
        # Queries = E_tokens [n_tokens × B × d], Keys/Values = frames_repr [T' × B × d]
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=nhead,
            dropout=multihead_dropout,
            batch_first=True,
        )

        self.norm_attn = nn.LayerNorm(hidden_size)
        
        # Proyección final por paso de tiempo
        self.proj = nn.Linear(hidden_size, output_size)

    def forward(self, x:torch.Tensor, frames_padding_mask:torch.Tensor) -> torch.Tensor:
        """
        x: Tensor of frames
        returns: Tensor of embeddings for each token (128 tokens of frames)
        """
        
        def transformer_checkpoint(x):
            return self.transformer(x, src_key_padding_mask=frames_padding_mask)

        B, T, N, C = x.shape                # x -> [batch_size, T, input_size]
        # print(f"Input shape: {x.shape}, Frames padding mask shape: {frames_padding_mask.shape}")
        x = x.permute(0, 3, 1, 2)           # [B, C, T, N]
        assert not torch.isnan(x).any(), "NaNs justo al iniciar"

        # print(f"Permuted input shape: {x.shape}")
        x = self.stgcn(x)                   # [B, 3*out_channels, T, K]
        # print(f"ST-GCN output shape: {x.shape}")
        assert not torch.isnan(x).any(), "NaNs después del ST-GCN"

        x = self.linear_hidden(x)           # [B, hidden, T, K]
        # print(f"Linear hidden output shape: {x.shape}")
        assert not torch.isnan(x).any(), "NaNs después de linear_hidden"

        x = x.mean(dim=-1)                  # [B, hidden, T]
        x = x.permute(0, 2, 1).contiguous() # [B, T, hidden]
        # print(f"Permuted linear hidden output shape: {x.shape}")
        assert not torch.isnan(x).any(), "NaNs después del pool nodos"

        if self.training:
            x = checkpoint(transformer_checkpoint, x, use_reentrant=False)
        else:
            x = transformer_checkpoint(x)  # [B, pool_dim, hidden]
        
        assert not torch.isnan(x).any(), "NaNs después del transformer"
        
        Q = self.token_queries.unsqueeze(0).expand(B, -1, -1)   # [B, n_tokens, output_size]
    
        attn_out, attn_w = self.cross_attn(
            query=Q,
            key=x,
            value=x,
            key_padding_mask=frames_padding_mask
        )  # [B, n_tokens, hidden]
        # print(f"Cross attention output shape: {attn_out.shape}, Attention weights shape: {attn_w.shape}")
        x = self.norm_attn(Q + attn_out)
        x = self.proj(x)     # [B, n_tokens, output_size]
        # print(f"Final output shape: {x.shape}")
        return x, attn_w