import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


@torch.jit.script
def imitator_loss(pred_embs: torch.Tensor, target_embs: torch.Tensor, embedding_mask: torch.Tensor = None) -> torch.Tensor:
    """
    Args:
        pred_embs: Tensor of shape (batch_size, seq_len, emb_dim)
        target_embs: Tensor of shape (batch_size, seq_len, emb_dim)
        embedding_mask: Tensor of shape (batch_size, seq_len) indicating valid positions (True for valid, False for invalid).
                        If None, all positions are considered valid.
    Returns:
        loss: Tensor of shape (batch_size, seq_len)
    """

    # align the lengths of the sequences
    L_common = min(pred_embs.size(1), target_embs.size(1))
    pred_embs     = pred_embs   [:, :L_common]
    target_embs   = target_embs [:, :L_common]
    embedding_mask = embedding_mask[:, :L_common]

    # Compute the L2 loss
    mse_per_element = F.mse_loss(pred_embs, target_embs, reduction='none')

    mse_per_token = mse_per_element.mean(dim=-1)

    valid = ~embedding_mask

    loss = (mse_per_token * valid).sum() / valid.sum()

    return loss


@torch.jit.script
def imitator_loss_masked(
    pred_embs: torch.Tensor,        # (B, T, D)
    target_embs: torch.Tensor,      # (B, T, D)
    embedding_mask: Optional[torch.Tensor],  # (B, T)
    ltm_dim: int = 2048,            # ← puedes ajustar a 1536, etc.
    alpha: float = 0.7
) -> torch.Tensor:
    """
    Calcula la pérdida MSE entre embeddings predichos y objetivo,
    separando el embedding en dos segmentos: LTM y HEAD.

    Args:
        pred_embs: Tensor (B, T, D) - embeddings predichos
        target_embs: Tensor (B, T, D) - embeddings de referencia
        embedding_mask: Tensor (B, T) - True para ignorar, False para incluir
        ltm_dim: Dimensiones dedicadas al segmento LTM
        alpha: Peso relativo de la pérdida en LTM vs HEAD

    Returns:
        loss: Escalar, pérdida promedio ponderada en tokens válidos
    """
    B, T, D = pred_embs.shape
    head_dim = D - ltm_dim
    assert head_dim > 0, "ltm_dim demasiado grande para la dimensión del embedding"

    # Asegura que secuencias tengan misma longitud
    L_common = min(pred_embs.size(1), target_embs.size(1))
    pred_embs = pred_embs[:, :L_common]
    target_embs = target_embs[:, :L_common]
    
    if embedding_mask is not None:
        embedding_mask = embedding_mask[:, :L_common]
    else:
        embedding_mask = torch.zeros((B, L_common), dtype=torch.bool, device=pred_embs.device)

    # Split LTM / HEAD
    pred_ltm, pred_head     = pred_embs[:, :, :ltm_dim], pred_embs[:, :, ltm_dim:]
    target_ltm, target_head = target_embs[:, :, :ltm_dim], target_embs[:, :, ltm_dim:]

    # MSE por token
    loss_ltm  = F.mse_loss(pred_ltm, target_ltm, reduction='none').mean(dim=-1)   # (B, T)
    loss_head = F.mse_loss(pred_head, target_head, reduction='none').mean(dim=-1)

    loss_per_token = alpha * loss_ltm + (1. - alpha) * loss_head

    valid = ~embedding_mask  # False = padding
    loss = (loss_per_token * valid).sum() / valid.sum().clamp(min=1)

    return loss
