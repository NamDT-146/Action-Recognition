import os
import sys
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.optim import lr_scheduler
import datetime
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from i3d.pytorch_i3d import InceptionI3d
from i3d import videotransforms
from test_dataset import ViolenceDataset
from ResNet3D.model import generate_model, load_pretrained_model


# Add regularization arguments to parser
parser = argparse.ArgumentParser()
parser.add_argument('-resume', type=str, default='', help='Path to checkpoint for resuming training')
parser.add_argument('-model', type=str, default='i3d', help='Model type: i3d or resnet')
parser.add_argument('-mode', type=str, default='rgb', help='rgb or flow')
parser.add_argument('-save_model', type=str, default='i3d_violence/models/')
parser.add_argument('-split_file', type=str, default='data/precomputed/split_info.csv')
parser.add_argument('-batch_size', type=int, default=16)
parser.add_argument('-num_frames', type=int, default=16)
parser.add_argument('-frame_size', type=int, default=224, help='Size of input frames (height and width)')
parser.add_argument('-logits_lr', type=float, default=0.01, help='Initial learning rate')
parser.add_argument('-backbone_lr', type=float, default=1e-4, help='Learning rate for backbone')
parser.add_argument('-max_steps', type=int, default=2048, help='Maximum training steps')
parser.add_argument('-steps_per_val', type=int, default=64, help='Steps between validations')
parser.add_argument('-steps_per_log', type=int, default=8, help='Steps between logging')
parser.add_argument('-warmup_steps', type=int, default=128, help='Number of warmup steps for learning rate')
parser.add_argument('-device', type=str, default='cuda', help='Device to use (e.g., cuda:0, cuda:1, cpu)')
parser.add_argument('-early_stopping', type=int, default=15, help='Early stopping patience (validation steps)')
parser.add_argument('-loss', type=str, default='bce_logits', help='Loss function: bce or focal')
parser.add_argument('-focal_gamma', type=float, default=2.0, help='Gamma parameter for focal loss')
parser.add_argument('-focal_alpha', type=float, default=None, help='Alpha parameter for focal loss')
# New regularization arguments
parser.add_argument('-weight_decay', type=float, default=1e-5, help='Weight decay (L2 regularization)')
parser.add_argument('-dropout', type=float, default=0.5, help='Dropout rate for regularization')
parser.add_argument('-label_smoothing', type=float, default=0.0, help='Label smoothing factor (0=disabled)')
parser.add_argument('-clip_grad', type=float, default=1.0, help='Gradient clipping norm (0=disabled)')

args = parser.parse_args()

def log_to_file(log_path, message):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, 'a') as f:
        f.write(f"[{timestamp}] {message}\n")
    print(message)  # Also print to console

def labels_to_onehot(labels, num_classes=2):
    batch_size = labels.size(0)
    onehot = torch.zeros(batch_size, num_classes, device=labels.device)
    onehot.scatter_(1, labels.unsqueeze(1), 1)
    return onehot

# Focal Loss with label smoothing support
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, reduction='mean', label_smoothing=0.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs, targets):
        # Apply label smoothing if enabled
        if self.label_smoothing > 0:
            # For one-hot encoded targets
            if targets.size() == inputs.size():
                num_classes = targets.size(1)
                # Smooth targets: move probability from 1.0 to uniform distribution
                targets = targets * (1 - self.label_smoothing) + self.label_smoothing / num_classes
        
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(inputs)
        
        # Calculate BCE loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Calculate focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma
        
        # Apply alpha if specified
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_weight = alpha_t * focal_weight
            
        # Calculate focal loss
        focal_loss = focal_weight * bce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def train_step(model, data, labels, optimizer, scheduler, criterion, device, 
               step, loss_type='bce_logits', label_smoothing=0.0, 
               clip_grad=1.0, log_path=None, writer=None, steps_to_log=10):

    # Move data to device
    data = data.to(device)
    labels = labels.to(device)
    
    # Forward pass
    per_frame_logits = model(data)
    
    # Reshape if necessary (I3D sometimes outputs [B, C, 1, 1, 1])
    if per_frame_logits.ndim > 2:
        per_frame_logits = per_frame_logits.squeeze(2)
    
    # Calculate loss using the provided criterion
    if loss_type == 'focal' or loss_type == 'bce_logits' or loss_type == 'bce':
        loss = criterion(per_frame_logits, labels)
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    
    # Backward pass and update weights (no accumulation)
    optimizer.zero_grad()
    loss.backward()
    
    # Apply gradient clipping if specified
    if clip_grad > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
    
    # Update weights
    optimizer.step()
    
    # Update learning rate
    scheduler.step()
    
    # Compute metrics
    with torch.no_grad():
        probs = torch.softmax(per_frame_logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        true_classes = torch.argmax(labels, dim=1)
        accuracy = (preds == true_classes).float().mean().item()
        
        # Log to TensorBoard
        if writer and step % steps_to_log == 0:
            writer.add_scalar('train/loss', loss.item(), step)
            writer.add_scalar('train/accuracy', accuracy, step)
            writer.add_scalar('lr/logits', optimizer.param_groups[1]['lr'], step)
            writer.add_scalar('lr/backbone', optimizer.param_groups[0]['lr'], step)
        
        # Log batch statistics
        if step % steps_to_log == 0:
            message = f"Step: {step}, Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}, " \
                      f"LR base: {optimizer.param_groups[0]['lr']:.6f}, " \
                      f"LR logits: {optimizer.param_groups[1]['lr']:.6f}"
            if log_path:
                log_to_file(log_path, message)
            else:
                print(message)
    
    return loss.item(), accuracy

def validate(model, val_loader, device, criterion, loss_type='bce_logits', 
             label_smoothing=0.0, log_path=None, writer=None, step=0):
    model.eval()
    val_losses = []
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for data, labels in val_loader:
            # Labels are assumed to already be one-hot encoded from main loop
            data = data.to(device)
            labels = labels.to(device)
            
            # Convert labels to one-hot encoding
            labels = labels_to_onehot(labels, num_classes=2)
            
            # Apply label smoothing if needed
            if label_smoothing > 0 and loss_type != 'focal':  # Focal loss handles smoothing internally
                labels = labels * (1 - label_smoothing) + label_smoothing / 2
            
            
            # Forward pass
            per_frame_logits = model(data)
            
            # Reshape if necessary
            if per_frame_logits.ndim > 2:
                per_frame_logits = per_frame_logits.squeeze(2)
            
            # Calculate loss using the provided criterion
            loss = criterion(per_frame_logits, labels)
            
            val_losses.append(loss.item())
            
            # Compute predictions
            probs = torch.softmax(per_frame_logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            true_classes = torch.argmax(labels, dim=1)
            
            all_preds.append(preds.cpu())
            all_true.append(true_classes.cpu())
    
    # Compute metrics
    all_preds = torch.cat(all_preds).numpy()
    all_true = torch.cat(all_true).numpy()
    val_accuracy = (all_preds == all_true).mean()
    val_loss = np.mean(val_losses)
    
    # Log metrics
    message = f"Validation at step {step}: Loss = {val_loss:.4f}, Accuracy = {val_accuracy:.4f}"
    if log_path:
        log_to_file(log_path, message)
    else:
        print(message)
    
    # Log to TensorBoard
    if writer:
        writer.add_scalar('val/loss', val_loss, step)
        writer.add_scalar('val/accuracy', val_accuracy, step)
    
    model.train()
    return val_loss, val_accuracy

def define_model(model_type, mode='rgb', dropout_rate=0.5, device='cuda', log_path=None):
    model = None
    if model_type == 'i3d':
        # Setup the model
        if mode == 'flow':
            model = InceptionI3d(400, in_channels=2, dropout_keep_prob=dropout_rate)
            model.load_state_dict(torch.load('pretrained_weight/i3d/models/flow_imagenet.pt'))
        else:
            model = InceptionI3d(400, in_channels=3, dropout_keep_prob=dropout_rate)
            model.load_state_dict(torch.load('pretrained_weight/i3d/models/rgb_imagenet.pt'))

        # Replace the last layer for binary classification
        model.replace_logits(2)

        # Apply dropout if needed
        if dropout_rate > 0:
            log_to_file(log_path, f"Adding dropout with rate {dropout_rate}")
            model.logits = nn.Sequential(
                nn.Dropout3d(dropout_rate),
                model.logits
            )

        model.to(device)
        
        # Freeze early layers
        freeze_layers = [
            'Conv3d_1a_7x7', 'Conv3d_2b_1x1', 'Conv3d_2c_3x3',
            'Mixed_3b', 'Mixed_3c',
            'Mixed_4b', 'Mixed_4c', 'Mixed_4d', 'Mixed_4e', 'Mixed_4f'
        ]

        log_to_file(log_path, "Freezing early layers...")
        for name, param in model.named_parameters():
            if any(layer in name for layer in freeze_layers):
                param.requires_grad = False
                log_to_file(log_path, f"  Freezing: {name}")
                               
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        log_to_file(log_path, f"Trainable/Total: {trainable_params}/{total_params}")

    elif model_type == 'resnet':
        class Opt:
            model = 'resnet'
            model_depth = 50
            n_classes = 1139  # Number of classes in pretraining
            n_input_channels = 3  # Number of input channels (e.g., RGB)
            resnet_shortcut = 'B'
            conv1_t_size = 7
            conv1_t_stride = 1
            no_max_pool = False
            resnet_widen_factor = 1.0
        opt = Opt()
        model = generate_model(opt)

        pretrain_path = 'pretrained_weight/resnet/r3d50_KMS_200ep.pth'
        model_name = 'resnet'
        n_finetune_classes = 2  # Set to your target number of classes

        model = load_pretrained_model(model, pretrain_path, model_name, n_finetune_classes)
        model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(model.fc.in_features, 2)
        )
        
        model.to(device)
        
        # Freeze early layers of ResNet
        log_to_file(log_path, "Freezing early layers of ResNet3D...")
        
        # Define which components to freeze (first convolutional layer and first residual block)
        freeze_components = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3']
        
        for name, param in model.named_parameters():
            if any(component in name for component in freeze_components):
                param.requires_grad = False
                log_to_file(log_path, f"  Freezing: {name}")
        
        # Log trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        log_to_file(log_path, f"Trainable/Total: {trainable_params}/{total_params}")
        
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return model

def define_loader(split_file, model_type='i3d', mode='rgb', num_frames=16, batch_size=16, 
                  val_split=0.2, random_seed=42, log_path=None):

    if model_type == 'i3d':
        train_transforms = videotransforms.Compose([
            videotransforms.RandomCrop(224),
            videotransforms.RandomHorizontalFlip(),
        ])

        test_transforms = videotransforms.Compose([
            videotransforms.CenterCrop(224),
        ])
    elif model_type == 'resnet':
        train_transforms = videotransforms.Compose([
            videotransforms.RandomCrop(224),
            videotransforms.Resize(128),
            videotransforms.RandomHorizontalFlip(),
        ])
        test_transforms = videotransforms.Compose([
            videotransforms.Resize(256),
            videotransforms.CenterCrop(224),
            videotransforms.Resize(128),
        ])

    # Load the complete training dataset
    full_train_dataset = ViolenceDataset(
        split_file=split_file,
        split='train',
        mode=mode,
        num_frames=num_frames,
        transforms=train_transforms,
        random_start=True
    )
    
    # Calculate sizes for the train-val split
    dataset_size = len(full_train_dataset)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    
    # Create the train-val split using random_split
    torch.manual_seed(random_seed)  # For reproducibility
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset, [train_size, val_size]
    )
    
    # Load the test dataset (unchanged)
    test_dataset = ViolenceDataset(
        split_file=split_file,
        split='test',
        mode=mode,
        num_frames=num_frames,
        transforms=test_transforms,
        random_start=False
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    # Log dataset information
    log_to_file(log_path, f"Dataset loaded with train-val-test split:")
    log_to_file(log_path, f"  Training samples: {train_size}")
    log_to_file(log_path, f"  Validation samples: {val_size}")
    log_to_file(log_path, f"  Test samples: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader

def evaluate_model(model, test_loader, device, criterion, loss_type='bce_logits', 
                  label_smoothing=0.0, log_path=None):
    model.eval()
    test_losses = []
    all_preds = []
    all_true = []
    
    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)
            
            # Convert labels to one-hot encoding
            labels = labels_to_onehot(labels, num_classes=2)
            
            # Apply label smoothing if needed
            if label_smoothing > 0 and loss_type != 'focal':
                labels = labels * (1 - label_smoothing) + label_smoothing / 2
            
            # Forward pass
            per_frame_logits = model(data)
            
            # Reshape if necessary
            if per_frame_logits.ndim > 2:
                per_frame_logits = per_frame_logits.squeeze(2).squeeze(2).squeeze(2)
            
            # Calculate loss
            loss = criterion(per_frame_logits, labels)
            test_losses.append(loss.item())
            
            # Compute predictions
            probs = torch.softmax(per_frame_logits, dim=1)
            preds = torch.argmax(probs, dim=1)
            true_classes = torch.argmax(labels, dim=1)
            
            all_preds.append(preds.cpu())
            all_true.append(true_classes.cpu())
    
    # Compute metrics
    all_preds = torch.cat(all_preds).numpy()
    all_true = torch.cat(all_true).numpy()
    test_accuracy = (all_preds == all_true).mean()
    test_loss = np.mean(test_losses)
    
    # Log metrics
    message = f"Test set evaluation: Loss = {test_loss:.4f}, Accuracy = {test_accuracy:.4f}"
    if log_path:
        log_to_file(log_path, message)
    else:
        print(message)
    
    return test_loss, test_accuracy

def run(mode='rgb', split_file='data/precomputed/split_info.csv', batch_size=16, 
        num_frames=64, logits_lr=0.01, backbone_lr=0.0001, max_steps=1024, save_model='models/', device='cuda',
        early_stopping_patience=5, loss_type='bce_logits', focal_gamma=2.0, focal_alpha=0.25,
        weight_decay=1e-5, dropout_rate=0.5, label_smoothing=0.1, clip_grad=1.0,
        steps_per_val=100, steps_to_log=10, warmup_steps=10000, val_split=0.2,
        resume_checkpoint=''):
    
    # Setup logging
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = os.path.join(os.path.dirname(save_model), f'runs_{timestamp}')
    log_path = os.path.join(log_dir, 'training_log.txt')
    os.makedirs(log_dir, exist_ok=True)
    
    # Initialize TensorBoard writer
    writer = SummaryWriter(log_dir=log_dir)
    
    # Log training parameters
    log_to_file(log_path, f"Starting training with parameters:")
    log_to_file(log_path, f"  Mode: {mode}")
    log_to_file(log_path, f"  Split file: {split_file}")
    log_to_file(log_path, f"  Validation split: {val_split}")
    log_to_file(log_path, f"  Batch size: {batch_size}")
    log_to_file(log_path, f"  Num frames: {num_frames}")
    log_to_file(log_path, f"  Device: {device}")
    log_to_file(log_path, f"  Early stopping patience: {early_stopping_patience}")
    log_to_file(log_path, f"  Loss function: {loss_type}")
    log_to_file(log_path, f"  Steps per validation: {steps_per_val}")
    log_to_file(log_path, f"  Steps to log: {steps_to_log}")
    log_to_file(log_path, f"  Warmup steps: {warmup_steps}")
    log_to_file(log_path, f"  Resume checkpoint: {resume_checkpoint if resume_checkpoint else 'None'}")
    
    # Log regularization parameters
    log_to_file(log_path, f"  Weight decay: {weight_decay}")
    log_to_file(log_path, f"  Dropout rate: {dropout_rate}")
    log_to_file(log_path, f"  Label smoothing: {label_smoothing}")
    log_to_file(log_path, f"  Gradient clipping: {clip_grad}")
    
    if loss_type == 'focal':
        log_to_file(log_path, f"  Focal gamma: {focal_gamma}")
        log_to_file(log_path, f"  Focal alpha: {focal_alpha}")
                    
    model = define_model(args.model, mode=mode, dropout_rate=dropout_rate, device=device, log_path=log_path)
    
    # Get train, validation, and test loaders with the new function
    train_loader, val_loader, test_loader = define_loader(
        split_file, 
        model_type=args.model, 
        mode=mode, 
        num_frames=num_frames, 
        batch_size=batch_size, 
        val_split=val_split,
        log_path=log_path
    )
    
    optimizer = optim.AdamW([
        {'params': [p for n, p in model.named_parameters() 
                    if 'logits' not in n and p.requires_grad], 
        'lr': backbone_lr},
        {'params': [p for n, p in model.named_parameters() 
                    if 'logits' in n], 
        'lr': logits_lr}
    ], weight_decay=weight_decay)
    
    # Learning rate scheduler with warmup
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=[
            # Backbone: Linear warmup followed by linear decay
            lambda step: min(1.0, step / warmup_steps) if step < warmup_steps else 
                max(0.0, 1.0 - (step - warmup_steps) / (max_steps - warmup_steps)),
            
            # Logits: Just linear decay (no warmup)
            lambda step: max(0.0, 1.0 - step / max_steps)
        ]
    )
    
    # Set up loss function
    if loss_type == 'focal':
        criterion = FocalLoss(gamma=focal_gamma, alpha=focal_alpha, label_smoothing=label_smoothing)
    elif loss_type == 'bce':
        criterion = nn.BCELoss()
    else:  # 'bce_logits'
        criterion = nn.BCEWithLogitsLoss()
    
    # Setup for early stopping and checkpointing
    best_val_accuracy = 0.0
    best_model_path = os.path.join(log_dir, 'best_model.pt')
    latest_model_path = os.path.join(log_dir, 'latest_model.pt')
    no_improvement_count = 0
    
    # Initialize step counter
    steps = 0
    
    # Check if we're resuming from a checkpoint
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        # Assume resume_checkpoint is a log directory path
        log_to_file(log_path, f"Resuming training from log directory: {resume_checkpoint}")
        
        # Construct the best model path from the log directory
        resume_model_path = os.path.join(resume_checkpoint, 'best_model.pt')
        
        if not os.path.exists(resume_model_path):
            # Try latest model if best model doesn't exist
            resume_model_path = os.path.join(resume_checkpoint, 'latest_model.pt')
            
        if not os.path.exists(resume_model_path):
            log_to_file(log_path, f"No model checkpoint found in {resume_checkpoint}. Starting from scratch.")
        else:
            log_to_file(log_path, f"Loading checkpoint from: {resume_model_path}")
            checkpoint = torch.load(resume_model_path, weights_only=False)
            
            # Restore model state
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Restore optimizer state
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            # Restore scheduler state if available
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            # Restore step counter and other training state
            steps = checkpoint.get('step', 0)
            
            # Restore best validation accuracy if available
            if 'val_accuracy' in checkpoint:
                best_val_accuracy = checkpoint['val_accuracy']
                
            log_to_file(log_path, f"Resumed training from step {steps} with validation accuracy: {best_val_accuracy:.4f}")
    
    # Main training loop
    while steps < max_steps:
        model.train()
        for data, labels in train_loader:
            # Convert labels to one-hot encoding
            labels_onehot = labels_to_onehot(labels, num_classes=2)
            
            # Apply label smoothing if needed
            if label_smoothing > 0 and loss_type != 'focal':  # Focal loss handles smoothing internally
                labels_onehot = labels_onehot * (1 - label_smoothing) + label_smoothing / 2
            
            # Train step
            loss, accuracy = train_step(
                model=model, 
                data=data, 
                labels=labels_onehot, 
                optimizer=optimizer, 
                scheduler=scheduler,
                criterion=criterion, 
                device=device, 
                step=steps, 
                loss_type=loss_type, 
                label_smoothing=label_smoothing, 
                clip_grad=clip_grad, 
                log_path=log_path, 
                writer=writer,
                steps_to_log=steps_to_log
            )           
            steps += 1
            
            # Validate periodically
            if steps % steps_per_val == 0:
                val_loss, val_accuracy = validate(
                    model=model,
                    val_loader=val_loader,
                    device=device,
                    criterion=criterion,
                    loss_type=loss_type,
                    label_smoothing=label_smoothing,
                    log_path=log_path,
                    writer=writer,
                    step=steps
                )
                
                # Save latest model
                torch.save({
                    'step': steps,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss,
                    'val_accuracy': val_accuracy,
                }, latest_model_path)
                
                # Check if it's the best model
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    # Save best model
                    torch.save({
                        'step': steps,
                        'model_state_dict': model.state_dict(),
                        'val_loss': val_loss,
                        'val_accuracy': val_accuracy,
                    }, best_model_path)
                    log_to_file(log_path, f"New best model saved with accuracy: {val_accuracy:.4f}")
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    if no_improvement_count >= early_stopping_patience:
                        log_to_file(log_path, f"Early stopping triggered after {no_improvement_count} validations without improvement")
                        break
            
            # Check if we've reached max steps
            if steps >= max_steps:
                break
        
        # Check for early stopping
        if no_improvement_count >= early_stopping_patience:
            break
    
    # Training complete - load the best model for final evaluation
    log_to_file(log_path, "Training completed. Loading best model for test evaluation...")
    
    # Load the best model
    checkpoint = torch.load(best_model_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate on the test set
    test_loss, test_accuracy = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        criterion=criterion,
        loss_type=loss_type,
        label_smoothing=label_smoothing,
        log_path=log_path
    )
    
    # Save test results with the best model
    checkpoint['test_loss'] = test_loss
    checkpoint['test_accuracy'] = test_accuracy
    torch.save(checkpoint, best_model_path)
    
    # Close TensorBoard writer
    writer.close()
    
    log_to_file(log_path, f"Training completed after {steps} steps")
    log_to_file(log_path, f"Best validation accuracy: {best_val_accuracy:.4f}")
    log_to_file(log_path, f"Test accuracy: {test_accuracy:.4f}")
    
    return best_val_accuracy, test_accuracy

if __name__ == '__main__':
    run(
        mode=args.mode,
        split_file=args.split_file,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        save_model=args.save_model,
        device=args.device,
        early_stopping_patience=args.early_stopping,
        max_steps=args.max_steps,
        steps_per_val=args.steps_per_val,
        steps_to_log=args.steps_per_log,
        warmup_steps=args.warmup_steps,
        logits_lr=args.logits_lr,
        backbone_lr=args.backbone_lr,
        loss_type=args.loss,
        focal_gamma=args.focal_gamma,
        focal_alpha=args.focal_alpha,
        weight_decay=args.weight_decay,
        dropout_rate=args.dropout,
        label_smoothing=args.label_smoothing,
        clip_grad=args.clip_grad,
        val_split=0.2,
        resume_checkpoint=args.resume
    )