import torch.nn as nn
import torch.nn.functional as F

class ImitatorLoss(nn.Module):

    def __init__(self):
        """
        L2 (MSE)
        return  L2
        """
        super().__init__()
    
    def forward(self, pred, target):
        """
        Compute the loss between predicted and target values.
        
        Args:
            pred (torch.Tensor): Predicted values.
            target (torch.Tensor): Target values.
        
        Returns:
            torch.Tensor: Computed loss value.
        """
        loss = F.mse_loss(pred, target)
        return loss