import torch.nn as nn
import torch.nn.functional as F

class ImitatorLoss(nn.Module):

    def __init__(self, alpha=1.0, beta=1.0):
        """
        Combina L2 (MSE) y CosineEmbeddingLoss:
        total_loss = alpha * L2 + beta * (1 - cos_similarity)
        """
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.cos_sim = nn.CosineSimilarity(dim=-1)

    def forward(self, pred, target):
        kl = F.kl_div(pred, target)
        return kl