"""
Training script for MIL Adapter models.
Trains lightweight adapter on precomputed X3D features for anomaly detection.
"""

import os
import sys
from pathlib import Path
import argparse
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset.loader.MILDataset import PrecomputedMILDataset
from model import get_adapter
from loss.mil_loss import get_mil_loss


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    
    losses = AverageMeter()
    ranking_losses = AverageMeter()
    reg_losses = AverageMeter()
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    
    for batch_idx, (features_normal, features_abnormal) in enumerate(pbar):
        # Move to device
        features_normal = features_normal.to(device)  # (B, S, C)
        features_abnormal = features_abnormal.to(device)
        
        # Forward pass
        scores_normal = model(features_normal)  # (B, S)
        scores_abnormal = model(features_abnormal)
        
        # Compute loss
        if isinstance(criterion, nn.Module):
            # Check if criterion returns multiple values
            loss_output = criterion(scores_normal, scores_abnormal)
            if isinstance(loss_output, tuple):
                loss, ranking_loss, reg_loss = loss_output
                ranking_losses.update(ranking_loss.item(), features_normal.size(0))
                reg_losses.update(reg_loss.item(), features_normal.size(0))
            else:
                loss = loss_output
                ranking_loss = loss
                reg_loss = torch.tensor(0.0)
        else:
            loss = criterion(scores_normal, scores_abnormal)
            ranking_loss = loss
            reg_loss = torch.tensor(0.0)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update metrics
        losses.update(loss.item(), features_normal.size(0))
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'rank': f'{ranking_losses.avg:.4f}' if ranking_losses.count > 0 else 'N/A',
            'reg': f'{reg_losses.avg:.4f}' if reg_losses.count > 0 else 'N/A'
        })
    
    return losses.avg, ranking_losses.avg, reg_losses.avg


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    
    losses = AverageMeter()
    all_scores_normal = []
    all_scores_abnormal = []
    
    for features_normal, features_abnormal in tqdm(dataloader, desc="Validating"):
        features_normal = features_normal.to(device)
        features_abnormal = features_abnormal.to(device)
        
        # Forward pass
        scores_normal = model(features_normal)
        scores_abnormal = model(features_abnormal)
        
        # Compute loss
        loss_output = criterion(scores_normal, scores_abnormal)
        if isinstance(loss_output, tuple):
            loss = loss_output[0]
        else:
            loss = loss_output
        
        losses.update(loss.item(), features_normal.size(0))
        
        # Store scores for analysis
        all_scores_normal.append(scores_normal.max(dim=1)[0].cpu())
        all_scores_abnormal.append(scores_abnormal.max(dim=1)[0].cpu())
    
    # Compute metrics
    scores_normal = torch.cat(all_scores_normal)
    scores_abnormal = torch.cat(all_scores_abnormal)
    
    # Simple accuracy: abnormal > normal
    correct = (scores_abnormal > scores_normal).float().mean().item()
    
    # Average max scores
    avg_normal = scores_normal.mean().item()
    avg_abnormal = scores_abnormal.mean().item()
    
    return losses.avg, correct, avg_normal, avg_abnormal


def plot_training_curves(history, save_dir):
    """Plot training curves"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[0, 1].plot(epochs, history['val_acc'], 'g-', label='Val Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Score comparison
    axes[1, 0].plot(epochs, history['avg_normal_score'], 'b-', label='Normal')
    axes[1, 0].plot(epochs, history['avg_abnormal_score'], 'r-', label='Abnormal')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Average Max Score')
    axes[1, 0].set_title('Score Comparison')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning rate
    if 'learning_rate' in history:
        axes[1, 1].plot(epochs, history['learning_rate'], 'purple')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_yscale('log')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to {save_dir / 'training_curves.png'}")


def save_checkpoint(model, optimizer, scheduler, epoch, best_acc, save_path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'best_acc': best_acc
    }
    torch.save(checkpoint, save_path)


def main(args):
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"{args.adapter_type}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print(f"Training MIL Adapter: {args.adapter_type.upper()}")
    print("="*80)
    print(f"Output directory: {output_dir}")
    
    # Save config
    config = vars(args)
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create datasets
    print("\nLoading datasets...")
    
    feature_augmentation = {
        'noise_std': args.noise_std,
        'dropout_rate': args.feature_dropout,
        'temporal_shift': args.temporal_shift,
        'temporal_dropout': args.temporal_dropout,
        'enable': args.use_augmentation
    }
    
    train_dataset = PrecomputedMILDataset(
        normal_features_dir=args.normal_features_dir,
        abnormal_features_dir=args.abnormal_features_dir,
        feature_augmentation=feature_augmentation,
        multi_view=args.multi_view,
        num_views=args.num_views,
        pca_file=args.pca_file,
        seed=args.seed
    )
    
    # For validation, use same dataset but disable augmentation
    val_feature_augmentation = feature_augmentation.copy()
    val_feature_augmentation['enable'] = False
    
    val_dataset = PrecomputedMILDataset(
        normal_features_dir=args.normal_features_dir,
        abnormal_features_dir=args.abnormal_features_dir,
        feature_augmentation=val_feature_augmentation,
        multi_view=args.multi_view,
        num_views=args.num_views,
        pca_file=args.pca_file,
        seed=args.seed + 1
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Get input dimension
    sample_normal, _ = train_dataset[0]
    input_dim = sample_normal.shape[1]
    print(f"Feature dimension: {input_dim}")
    
    # Create model
    print(f"\nCreating {args.adapter_type.upper()} adapter...")
    model = get_adapter(
        adapter_type=args.adapter_type,
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout
    )
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Loss function
    print(f"\nUsing loss: {args.loss_type}")
    if args.loss_type == 'ranking':
        criterion = get_mil_loss('ranking', margin=args.margin)
    elif args.loss_type == 'ranking_sparsity':
        criterion = get_mil_loss('ranking_sparsity', margin=args.margin, sparsity_weight=args.sparsity_weight)
    elif args.loss_type == 'ranking_smoothing':
        criterion = get_mil_loss('ranking_smoothing', margin=args.margin, smoothing_weight=args.smoothing_weight)
    else:
        raise ValueError(f"Unknown loss type: {args.loss_type}")
    
    # Optimizer
    if args.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")
    
    # Scheduler
    if args.scheduler == 'plateau':
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    elif args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    else:
        scheduler = None
    
    # Training loop
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80)
    
    best_acc = 0.0
    best_loss = float('inf')
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': [],
        'avg_normal_score': [],
        'avg_abnormal_score': [],
        'learning_rate': []
    }
    
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 60)
        
        # Train
        train_loss, train_ranking, train_reg = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc, avg_normal, avg_abnormal = validate(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['avg_normal_score'].append(avg_normal)
        history['avg_abnormal_score'].append(avg_abnormal)
        history['learning_rate'].append(current_lr)
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        print(f"  Val Accuracy: {val_acc:.4f}")
        print(f"  Avg Normal Score: {avg_normal:.4f}")
        print(f"  Avg Abnormal Score: {avg_abnormal:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(model, optimizer, scheduler, epoch, best_acc, 
                          output_dir / 'best_model_acc.pth')
            print(f"  ✓ Saved best accuracy model: {best_acc:.4f}")
        
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(model, optimizer, scheduler, epoch, best_acc,
                          output_dir / 'best_model_loss.pth')
            print(f"  ✓ Saved best loss model: {best_loss:.6f}")
        
        # Save latest model
        if epoch % args.save_every == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, best_acc,
                          output_dir / f'checkpoint_epoch_{epoch}.pth')
    
    # Save final model
    save_checkpoint(model, optimizer, scheduler, args.epochs, best_acc,
                   output_dir / 'final_model.pth')
    
    # Plot training curves
    plot_training_curves(history, output_dir)
    
    # Save history
    np.save(output_dir / 'history.npy', history)
    
    print("\n" + "="*80)
    print("Training Complete!")
    print("="*80)
    print(f"Best Validation Accuracy: {best_acc:.4f}")
    print(f"Best Validation Loss: {best_loss:.6f}")
    print(f"Models saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MIL Adapter for Anomaly Detection")
    
    # Data
    parser.add_argument('--normal_features_dir', type=str, required=True,
                       help='Path to normal features directory')
    parser.add_argument('--abnormal_features_dir', type=str, required=True,
                       help='Path to abnormal features directory')
    parser.add_argument('--pca_file', type=str, default=None,
                       help='Path to PCA transform file')
    parser.add_argument('--multi_view', action='store_true',
                       help='Use multi-view features')
    parser.add_argument('--num_views', type=int, default=5,
                       help='Number of views per video (if multi_view)')
    
    # Model
    parser.add_argument('--adapter_type', type=str, default='lstm',
                       choices=['mlp', 'lstm', 'conv1d'],
                       help='Type of adapter model')
    parser.add_argument('--hidden_dim', type=int, default=32,
                       help='Hidden dimension size')
    parser.add_argument('--dropout', type=float, default=0.6,
                       help='Dropout rate')
    
    # Loss
    parser.add_argument('--loss_type', type=str, default='ranking',
                       choices=['ranking', 'ranking_sparsity', 'ranking_smoothing'],
                       help='Loss function type')
    parser.add_argument('--margin', type=float, default=1.0,
                       help='Margin for ranking loss')
    parser.add_argument('--sparsity_weight', type=float, default=0.01,
                       help='Weight for sparsity loss')
    parser.add_argument('--smoothing_weight', type=float, default=0.01,
                       help='Weight for smoothing loss')
    
    # Augmentation
    parser.add_argument('--use_augmentation', action='store_true',
                       help='Use feature augmentation')
    parser.add_argument('--noise_std', type=float, default=0.05,
                       help='Standard deviation of Gaussian noise')
    parser.add_argument('--feature_dropout', type=float, default=0.1,
                       help='Feature dropout rate')
    parser.add_argument('--temporal_shift', type=int, default=2,
                       help='Maximum temporal shift')
    parser.add_argument('--temporal_dropout', type=float, default=0.2,
                       help='Temporal dropout rate')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'adamw'],
                       help='Optimizer')
    parser.add_argument('--weight_decay', type=float, default=0.0005,
                       help='Weight decay')
    parser.add_argument('--scheduler', type=str, default='plateau',
                       choices=['plateau', 'cosine', 'none'],
                       help='Learning rate scheduler')
    
    # System
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/mil_adapter',
                       help='Output directory')
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    main(args)