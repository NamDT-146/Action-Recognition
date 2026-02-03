"""
Multiple Instance Learning (MIL) Loss Functions for Anomaly Detection.
Implements ranking loss variants from Sultani et al.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MILRankingLoss(nn.Module):
    """
    Standard MIL Ranking Loss for weakly-supervised anomaly detection.
    
    Loss = max(0, 1 - max(abnormal_scores) + max(normal_scores))
    
    This loss encourages:
    - Maximum abnormal score > Maximum normal score by a margin of 1
    - The model to assign high scores to at least one segment in abnormal videos
    - The model to assign low scores to all segments in normal videos
    """
    
    def __init__(self, margin: float = 1.0):
        """
        Args:
            margin: Margin for ranking loss
        """
        super().__init__()
        self.margin = margin
    
    def forward(self, scores_normal: torch.Tensor, scores_abnormal: torch.Tensor) -> torch.Tensor:
        """
        Compute MIL ranking loss.
        
        Args:
            scores_normal: Normal video scores (Batch, Segments)
            scores_abnormal: Abnormal video scores (Batch, Segments)
            
        Returns:
            loss: Scalar loss value
        """
        # Get maximum score for each video
        max_normal = torch.max(scores_normal, dim=1)[0]  # (Batch,)
        max_abnormal = torch.max(scores_abnormal, dim=1)[0]  # (Batch,)
        
        # Ranking loss: encourage max_abnormal > max_normal + margin
        loss = torch.mean(F.relu(self.margin - max_abnormal + max_normal))
        
        return loss


class MILRankingLossWithSparsity(nn.Module):
    """
    MIL Ranking Loss with sparsity regularization.
    
    Total Loss = Ranking Loss + λ * Sparsity Loss
    
    Sparsity loss encourages the model to:
    - Assign high scores to only a few segments (sparse anomaly detection)
    - Not predict every segment as anomalous
    """
    
    def __init__(self, margin: float = 1.0, sparsity_weight: float = 0.01):
        """
        Args:
            margin: Margin for ranking loss
            sparsity_weight: Weight for sparsity regularization
        """
        super().__init__()
        self.margin = margin
        self.sparsity_weight = sparsity_weight
    
    def forward(self, scores_normal: torch.Tensor, scores_abnormal: torch.Tensor) -> tuple:
        """
        Compute MIL ranking loss with sparsity.
        
        Args:
            scores_normal: Normal video scores (Batch, Segments)
            scores_abnormal: Abnormal video scores (Batch, Segments)
            
        Returns:
            total_loss: Total loss
            ranking_loss: Ranking loss component
            sparsity_loss: Sparsity loss component
        """
        # Ranking loss
        max_normal = torch.max(scores_normal, dim=1)[0]
        max_abnormal = torch.max(scores_abnormal, dim=1)[0]
        ranking_loss = torch.mean(F.relu(self.margin - max_abnormal + max_normal))
        
        # Sparsity loss: encourage most scores to be small
        # Use L1 norm on abnormal scores
        sparsity_loss = torch.mean(scores_abnormal)
        
        # Total loss
        total_loss = ranking_loss + self.sparsity_weight * sparsity_loss
        
        return total_loss, ranking_loss, sparsity_loss


class MILRankingLossWithSmoothing(nn.Module):
    """
    MIL Ranking Loss with temporal smoothing.
    
    Total Loss = Ranking Loss + λ * Smoothing Loss
    
    Smoothing loss encourages temporal consistency:
    - Adjacent segments should have similar scores
    - Prevents erratic score fluctuations
    """
    
    def __init__(self, margin: float = 1.0, smoothing_weight: float = 0.01):
        """
        Args:
            margin: Margin for ranking loss
            smoothing_weight: Weight for temporal smoothing
        """
        super().__init__()
        self.margin = margin
        self.smoothing_weight = smoothing_weight
    
    def forward(self, scores_normal: torch.Tensor, scores_abnormal: torch.Tensor) -> tuple:
        """
        Compute MIL ranking loss with temporal smoothing.
        
        Args:
            scores_normal: Normal video scores (Batch, Segments)
            scores_abnormal: Abnormal video scores (Batch, Segments)
            
        Returns:
            total_loss: Total loss
            ranking_loss: Ranking loss component
            smoothing_loss: Smoothing loss component
        """
        # Ranking loss
        max_normal = torch.max(scores_normal, dim=1)[0]
        max_abnormal = torch.max(scores_abnormal, dim=1)[0]
        ranking_loss = torch.mean(F.relu(self.margin - max_abnormal + max_normal))
        
        # Temporal smoothing loss: penalize large differences between adjacent segments
        # Compute differences: |score[t] - score[t-1]|
        diff_normal = torch.abs(scores_normal[:, 1:] - scores_normal[:, :-1])
        diff_abnormal = torch.abs(scores_abnormal[:, 1:] - scores_abnormal[:, :-1])
        
        smoothing_loss = torch.mean(diff_normal) + torch.mean(diff_abnormal)
        
        # Total loss
        total_loss = ranking_loss + self.smoothing_weight * smoothing_loss
        
        return total_loss, ranking_loss, smoothing_loss


def get_mil_loss(loss_type: str = 'ranking', **kwargs):
    """
    Factory function to get MIL loss function.
    
    Args:
        loss_type: 'ranking', 'ranking_sparsity', or 'ranking_smoothing'
        **kwargs: Additional arguments for loss function
        
    Returns:
        Loss function instance
    """
    loss_type = loss_type.lower()
    
    if loss_type == 'ranking':
        return MILRankingLoss(**kwargs)
    elif loss_type == 'ranking_sparsity':
        return MILRankingLossWithSparsity(**kwargs)
    elif loss_type == 'ranking_smoothing':
        return MILRankingLossWithSmoothing(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# Test
if __name__ == "__main__":
    print("="*80)
    print("Testing MIL Loss Functions")
    print("="*80)
    
    # Test data
    batch_size = 8
    num_segments = 5
    
    # Simulate scores
    scores_normal = torch.rand(batch_size, num_segments) * 0.3  # Lower scores
    scores_abnormal = torch.rand(batch_size, num_segments) * 0.7 + 0.3  # Higher scores
    
    print(f"\nNormal scores shape: {scores_normal.shape}")
    print(f"Normal scores range: [{scores_normal.min():.4f}, {scores_normal.max():.4f}]")
    print(f"Abnormal scores shape: {scores_abnormal.shape}")
    print(f"Abnormal scores range: [{scores_abnormal.min():.4f}, {scores_abnormal.max():.4f}]")
    
    # Test each loss
    print("\n" + "="*60)
    print("Testing MILRankingLoss")
    print("="*60)
    loss_fn = MILRankingLoss(margin=1.0)
    loss = loss_fn(scores_normal, scores_abnormal)
    print(f"Loss: {loss.item():.6f}")
    
    print("\n" + "="*60)
    print("Testing MILRankingLossWithSparsity")
    print("="*60)
    loss_fn = MILRankingLossWithSparsity(margin=1.0, sparsity_weight=0.01)
    total_loss, ranking_loss, sparsity_loss = loss_fn(scores_normal, scores_abnormal)
    print(f"Total Loss: {total_loss.item():.6f}")
    print(f"Ranking Loss: {ranking_loss.item():.6f}")
    print(f"Sparsity Loss: {sparsity_loss.item():.6f}")
    
    print("\n" + "="*60)
    print("Testing MILRankingLossWithSmoothing")
    print("="*60)
    loss_fn = MILRankingLossWithSmoothing(margin=1.0, smoothing_weight=0.01)
    total_loss, ranking_loss, smoothing_loss = loss_fn(scores_normal, scores_abnormal)
    print(f"Total Loss: {total_loss.item():.6f}")
    print(f"Ranking Loss: {ranking_loss.item():.6f}")
    print(f"Smoothing Loss: {smoothing_loss.item():.6f}")
    
    print("\n" + "="*80)
    print("All loss tests passed!")
    print("="*80)