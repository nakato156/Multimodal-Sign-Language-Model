import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenwiseEmbeddingLoss(nn.Module):
    """
    L2 loss token por token entre embeddings predichos y embeddings reales.
    Ambos tensores deben tener forma [B, L, D], donde:
      - B: batch size
      - L: longitud de secuencia (tokens)
      - D: dimensión del embedding (e.g. 4096 en LLaMA 7B)
    """

    def __init__(self, reduction: str = 'mean'):
        """
        reduction: 'mean' → devuelve promedio total
                   'sum' → devuelve suma total
                   'none' → devuelve tensor de shape [B, L]
        """
        super().__init__()
        assert reduction in {'mean', 'sum', 'none'}
        self.reduction = reduction

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        """
        pred:    [B, L, D] – embeddings generados por el Motion Encoder
        target:  [B, L, D] – embeddings reales de LLaMA para los tokens
        """
        assert pred.shape == target.shape, "Shape mismatch"
        loss = F.mse_loss(pred, target, reduction='none')  # [B, L, D]
        loss = loss.mean(dim=-1)  # [B, L], promedio por token

        if self.reduction == 'mean':
            return loss.mean()   # escalar
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss          # shape [B, L]


class ImitatorLoss(nn.Module):
    def __init__(self, use_ce=False, lm_head=None, ce_weight=1.0):
        """
        L2 loss (token-wise) + optional cross-entropy via LM head.

        Args:
            use_ce (bool): Si se usa CrossEntropy adicional.
            lm_head (nn.Module): lm_head de LLaMA (congelado).
            vocab_size (int): Tamaño del vocabulario (necesario si ce=True).
            ce_weight (float): Peso relativo de la pérdida CE.
        """
        super().__init__()
        self.use_ce = use_ce
        self.ce_weight = ce_weight
        self.lm_head = lm_head
        self.l2_loss = TokenwiseEmbeddingLoss(reduction='mean')
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=128004)
        
        if use_ce and lm_head is None:
            raise ValueError("lm_head is required if use_ce is True.")

    def forward(self, pred_embs: torch.Tensor, target_embs: torch.Tensor, target_ids: torch.Tensor=None):
        """
        pred_embs: [B, L, D] – embeddings generados desde keypoints
        target_embs: [B, L, D] – embeddings reales de LLaMA
        target_ids: [B, L] – ids de tokens para supervisar CE

        Returns:
          total_loss: escalar
        """
        loss_l2: torch.Tensor = self.l2_loss(pred_embs, target_embs)

        if self.use_ce and target_ids is not None:
            assert target_ids.shape == (pred_embs.shape[0], pred_embs.shape[1]), \
                "target_ids must have shape [B, L] matching pred_embs and target_embs"
            logits = self.lm_head(pred_embs)  # [B, L, V]
            loss_ce: torch.Tensor = self.ce_loss(logits.view(-1, logits.size(-1)),
                                                  target_ids.view(-1)).to(pred_embs.device)
        else:
            loss_ce = torch.Tensor([0]).to(pred_embs.device)

        total_loss = loss_l2 + self.ce_weight * loss_ce
        return total_loss
