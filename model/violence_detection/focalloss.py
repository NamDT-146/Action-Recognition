import torch
from torch import nn
import torch.nn.functional as F

# Add this after your imports
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='sum'):
        """
        Focal Loss implementation for multi-class classification
        
        Args:
            alpha: Weighting factor to balance positive vs negative examples (0.25 is good for imbalanced datasets)
            gamma: Focusing parameter - higher gamma gives more weight to hard examples
            reduction: 'mean', 'sum' or 'none'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = 1e-6  # Small constant to prevent numerical instability
        
    def forward(self, inputs, targets):
        """
        Calculate focal loss
        
        Args:
            inputs: Model outputs (logits, before softmax)
            targets: One-hot encoded target labels
        """
        # Get softmax probabilities
        probs = F.softmax(inputs, dim=1)
        
        # # Convert targets to one-hot if they aren't already
        if targets.dim() == 1:
            targets = F.one_hot(targets, num_classes=inputs.size(1))
        
        # Calculate focal weight
        pt = torch.sum(targets * probs, dim=1)  # Probability of the true class
        focal_weight = (1 - pt) ** self.gamma
                
        # Apply alpha weighting for class balance
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            alpha_t = alpha_t.T
            focal_weight = alpha_t.squeeze() * focal_weight
            
        # Calculate cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets.float(), reduction='none')
        
        # Apply focal weighting to the loss
        loss = focal_weight * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss