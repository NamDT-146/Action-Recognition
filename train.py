import os
import yaml
import argparse
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from datetime import datetime
import numpy as np
from tqdm import tqdm
from pathlib import Path
from dotenv import load_dotenv
import socket
import time

# Import MLflow
import mlflow
import mlflow.pytorch

# Import TensorBoard
from torch.utils.tensorboard import SummaryWriter

# Import custom modules
from dataset import get_data_loader
from model import get_model

# Load environment variables
load_dotenv()


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def check_mlflow_connection(tracking_uri, timeout=5):
    """
    Check if MLflow server is reachable
    
    Args:
        tracking_uri (str): MLflow tracking URI (e.g., http://192.168.1.198:5231)
        timeout (int): Connection timeout in seconds
        
    Returns:
        bool: True if MLflow is reachable, False otherwise
    """
    try:
        # Extract host and port from URI
        if tracking_uri.startswith('http://'):
            uri = tracking_uri[7:]
        elif tracking_uri.startswith('https://'):
            uri = tracking_uri[8:]
        else:
            return False
        
        host, port = uri.split(':')
        port = int(port)
        
        # Try to connect
        socket.create_connection((host, port), timeout=timeout)
        return True
    except (socket.timeout, socket.error, ValueError, OSError):
        return False
    
def freeze_layers(model, num_layers_to_freeze):
    """
    Freeze first N layers of the model.
    
    Args:
        model: PyTorch model
        num_layers_to_freeze: Number of layers to freeze from the beginning
    """
    frozen_count = 0
    for name, param in model.named_parameters():
        if frozen_count < num_layers_to_freeze:
            param.requires_grad = False
            frozen_count += 1
        else:
            break
    
    print(f"Froze {frozen_count} layers")


def unfreeze_all_layers(model):
    """Unfreeze all layers in the model."""
    for param in model.parameters():
        param.requires_grad = True
    print("All layers unfrozen")


def get_trainable_params_count(model):
    """Get count of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_optimizer(model, config):
    """Create optimizer based on config"""
    optimizer_type = config.get('optimizer', 'adam').lower()
    lr = config.get('lr', 1e-4)
    weight_decay = config.get('weight_decay', 0.0)
    
    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        momentum = config.get('momentum', 0.9)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_type == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")
    
    return optimizer


def get_scheduler(optimizer, config):
    """Create learning rate scheduler based on config"""
    scheduler_type = config.get('scheduler', 'step').lower()
    
    if scheduler_type == 'step':
        step_size = config.get('scheduler_step_size', 10)
        gamma = config.get('scheduler_gamma', 0.1)
        scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'cosine':
        T_max = config.get('epochs', 50)
        scheduler = CosineAnnealingLR(optimizer, T_max=T_max)
    elif scheduler_type == 'none':
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_type}")
    
    return scheduler


def get_criterion(config):
    """Create loss function based on config"""
    loss_type = config.get('loss_type', 'bce').lower()
    
    if loss_type == 'bce':
        criterion = nn.BCELoss()
    elif loss_type == 'bce_logits':
        criterion = nn.BCEWithLogitsLoss()
    elif loss_type == 'crossentropy':
        criterion = nn.CrossEntropyLoss()
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    
    return criterion


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, config, writer=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]')
    for batch_idx, batch_data in enumerate(pbar):
        # Handle different dataset outputs
        if len(batch_data) == 3:
            sequences, labels, _ = batch_data  # Hand gestures with metadata
        else:
            sequences, labels = batch_data  # Video dataset
        
        sequences = sequences.to(device).float()
        labels = labels.to(device)
        
        # Adjust labels for different loss types
        if isinstance(criterion, nn.BCELoss):
            labels = labels.float().view(-1, 1)
        elif isinstance(criterion, nn.BCEWithLogitsLoss):
            labels = labels.float().view(-1, 1)
        elif isinstance(criterion, nn.CrossEntropyLoss):
            labels = labels.long().squeeze()  # Ensure labels are 1D
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(sequences)
        
        if len(outputs.shape) == 3 and outputs.shape[2] == 1:
            outputs = outputs.squeeze(2)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        if config.get('clip_gradient') is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_gradient'])
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        
        # Calculate accuracy
        if isinstance(criterion, nn.BCELoss) or isinstance(criterion, nn.BCEWithLogitsLoss):
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
        else:  # CrossEntropyLoss
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
        
        total += labels.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
        
        # Log to TensorBoard if available
        if writer is not None:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/BatchLoss', loss.item(), global_step)
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device, epoch, writer=None):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]')
        for batch_idx, batch_data in enumerate(pbar):
            # Handle different dataset outputs
            if len(batch_data) == 3:
                sequences, labels, _ = batch_data
            else:
                sequences, labels = batch_data
            
            sequences = sequences.to(device).float()
            labels = labels.to(device)
            
            # Adjust labels for different loss types
            if isinstance(criterion, nn.BCELoss):
                labels = labels.float().view(-1, 1)
            elif isinstance(criterion, nn.BCEWithLogitsLoss):
                labels = labels.float().view(-1, 1)
            elif isinstance(criterion, nn.CrossEntropyLoss):
                labels = labels.long().squeeze()  # Ensure labels are 1D
            
            # Forward pass
            outputs = model(sequences)
            if len(outputs.shape) == 3 and outputs.shape[2] == 1:
                outputs = outputs.squeeze(2)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            
            # Calculate accuracy
            if isinstance(criterion, nn.BCELoss) or isinstance(criterion, nn.BCEWithLogitsLoss):
                predicted = (outputs > 0.5).float()
                correct += (predicted == labels).sum().item()
            else:  # CrossEntropyLoss
                _, predicted = torch.max(outputs.data, 1)
                correct += (predicted == labels).sum().item()
            
            total += labels.size(0)
            
            pbar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': 100. * correct / total
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc

def save_checkpoint(state, save_dir, filename='checkpoint.pth'):
    """Save model checkpoint"""
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)
    print(f"Checkpoint saved to {filepath}")


def main(config_path):
    # Load configuration
    config = load_config(config_path)
    print("Configuration loaded:")
    print(yaml.dump(config, default_flow_style=False))
    
    # Set random seed
    set_seed(config.get('seed', 42))
    
    # Set device
    device = torch.device(f"cuda:{config.get('gpu', 0)}" if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create save directory
    save_dir = config.get('save_dir', './checkpoints')
    save_name = config.get('save_name', 'experiment')
    full_save_dir = os.path.join(save_dir, save_name)
    os.makedirs(full_save_dir, exist_ok=True)
    
    # Check MLflow connection
    mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://192.168.1.198:5231')
    mlflow_timeout = config.get('mlflow_timeout', 5)
    use_mlflow = False
    writer = None
    
    print(f"Checking MLflow connection to {mlflow_tracking_uri} (timeout: {mlflow_timeout}s)...")
    if check_mlflow_connection(mlflow_tracking_uri, timeout=mlflow_timeout):
        print("✅ MLflow connection successful!")
        use_mlflow = True
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        mlflow.set_experiment(config.get('experiment_name', 'action_recognition'))
    else:
        print(f"❌ MLflow connection failed. Falling back to TensorBoard...")
        use_mlflow = False
        # Setup TensorBoard
        writer = SummaryWriter(os.path.join(full_save_dir, 'runs'))
    
    # Context manager for MLflow (or dummy if not available)
    class DummyMlflowRun:
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            pass
    
    mlflow_context = mlflow.start_run(run_name=save_name) if use_mlflow else DummyMlflowRun()
    
    with mlflow_context:
        # Log all config parameters
        if use_mlflow:
            mlflow.log_params(config)
        
        # Create data loaders
        print("Creating data loaders...")
        train_loader, val_loader, test_loader = get_data_loader(
            data_dir=config.get('data_dir', '/home/atin-ct3/action_recognition/data'),
            dataset_name=config.get('dataset', 'hand-gestures'),
            batch_size=config.get('batch_size', 4),
            figure_size=config.get('img_size', 224),
            seq_length=config.get('seq_length', 20),
            crop_dark=config.get('crop_dark', None),
            num_workers=config.get('num_workers', 4),
            train_split=config.get('train_split', 0.7),
            val_split=config.get('val_split', 0.15),
            gesture_filter=config.get('gesture_filter', None),
            set_filter=config.get('set_filter', None),
            model_name=config.get('model_name', 'LSTM_CNN')
        )
        
        # Create model
        print("Creating model...")
        model_name = config.get('model_name', 'LSTM_CNN')
        
        if model_name == 'TimeSformer':
        # Create TimeSformer model
            model = get_model(**config)
            optimizer = get_optimizer(model, config)
            criterion = get_criterion(config)
        if model_name == 'LSTM_CNN':
            model, optimizer, criterion = get_model(**config)
            criterion = get_criterion(config)
        else:
            # For other models like TSM, create manually
            model = get_model(**config)
            optimizer = get_optimizer(model, config)
            criterion = get_criterion(config)
        
        model = model.to(device)
        
        # Create scheduler
        scheduler = get_scheduler(optimizer, config)

        if config.get('pretrained_weight', ""):
            pretrained_path = config['pretrained_weight']
            if os.path.exists(pretrained_path):
                print(f"Loading pretrained weights from: {pretrained_path}")
                pretrained_dict = torch.load(pretrained_path, map_location=device)
                
                # Handle mismatch in the last layer (logits) due to different number of classes
                # Remove keys related to the classification head to avoid size mismatch
                keys_to_remove = [k for k in pretrained_dict.keys() if 'logits' in k]
                for k in keys_to_remove:
                    del pretrained_dict[k]
                
                model.load_state_dict(pretrained_dict, strict=False)
                print(f"Loaded pretrained weights (excluding mismatched logits layer)")
                
                # Freeze layers if pretrained
                num_layers_to_freeze = config.get('num_layers_to_freeze', 0)
                if num_layers_to_freeze > 0:
                    freeze_layers(model, num_layers_to_freeze)
                
                warmup_epochs = config.get('warmup_epochs', 0)
                if warmup_epochs > 0:
                    print(f"Warmup will be applied for {warmup_epochs} epochs")
            else:
                print(f"Warning: Pretrained weights path does not exist: {pretrained_path}")
        else:
            warmup_epochs = 0

        # Resume from checkpoint if specified
        start_epoch = 0
        best_val_acc = 0.0
        if config.get('resume', False) and config.get('load_path', None):
            load_path = config['load_path']
            if os.path.exists(load_path):
                print(f"Resuming from checkpoint: {load_path}")
                checkpoint = torch.load(load_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch']
                best_val_acc = checkpoint.get('best_val_acc', 0.0)
                print(f"Resumed from epoch {start_epoch}")
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        if use_mlflow:
            mlflow.log_param("total_params", total_params)
            mlflow.log_param("trainable_params", trainable_params)
        
        # Training loop
        epochs = config.get('epochs', 50)
        warmup_epochs = config.get('warmup_epochs', 0)
        num_layers_to_freeze = config.get('num_layers_to_freeze', 0)
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Using logging: {'MLflow' if use_mlflow else 'TensorBoard'}")
        
        for epoch in range(start_epoch, epochs):
            # Unfreeze layers after warmup
            if epoch == warmup_epochs and num_layers_to_freeze > 0 and warmup_epochs > 0:
                print(f"\n🔓 Unfreezing all layers after warmup (epoch {epoch})")
                unfreeze_all_layers(model)
                
                # Optionally reset optimizer for fine-tuning phase
                if config.get('reset_optimizer_after_warmup', False):
                    print("Resetting optimizer for fine-tuning phase")
                    optimizer = get_optimizer(model, config)
                    scheduler = get_scheduler(optimizer, config)
            
            # Train
            train_loss, train_acc = train_epoch(
                model, train_loader, criterion, optimizer, device, epoch, config, writer
            )
            
            # Validate
            val_loss, val_acc = validate(
                model, val_loader, criterion, device, epoch, writer
            )
            
            # Update scheduler
            if scheduler is not None:
                scheduler.step()
            
            # Log metrics
            print(f"\nEpoch {epoch}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Log trainable parameters count during warmup
            if epoch < warmup_epochs:
                trainable_count = get_trainable_params_count(model)
                print(f"[Warmup Phase] Trainable parameters: {trainable_count:,}")
            
            if use_mlflow:
                # Log to MLflow
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("train_acc", train_acc, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
                mlflow.log_metric("val_acc", val_acc, step=epoch)
                mlflow.log_metric("lr", optimizer.param_groups[0]['lr'], step=epoch)
                mlflow.log_metric("trainable_params", get_trainable_params_count(model), step=epoch)
            else:
                # Log to TensorBoard
                writer.add_scalar('Train/Loss', train_loss, epoch)
                writer.add_scalar('Train/Accuracy', train_acc, epoch)
                writer.add_scalar('Val/Loss', val_loss, epoch)
                writer.add_scalar('Val/Accuracy', val_acc, epoch)
                writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
                writer.add_scalar('Trainable_Params', get_trainable_params_count(model), epoch)
            
            # Save checkpoint
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'best_val_acc': best_val_acc,
                'config': config
            }
            
            # Save latest checkpoint
            save_checkpoint(checkpoint, full_save_dir, 'latest_model.pth')
            
            # Save best checkpoint
            if is_best:
                save_checkpoint(checkpoint, full_save_dir, 'best_model.pth')
                print(f"New best model saved with validation accuracy: {val_acc:.2f}%")
                
                if use_mlflow:
                    # Log best model artifact to MLflow
                    mlflow.log_artifact(os.path.join(full_save_dir, 'best_model.pth'))
        
        # Final evaluation on test set
        print("\nEvaluating on test set...")
        test_loss, test_acc = validate(model, test_loader, criterion, device, epochs-1, writer)
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
        
        if use_mlflow:
            # Log final test metrics
            mlflow.log_metric("test_loss", test_loss)
            mlflow.log_metric("test_acc", test_acc)
        
        # Create dummy input for model signature
        dummy_batch = next(iter(test_loader))
        if len(dummy_batch) == 3:
            dummy_input, _, _ = dummy_batch
        else:
            dummy_input, _ = dummy_batch
        dummy_input = dummy_input[:1].cpu().numpy()  # Take one sample and convert to numpy
        
        # Log final model to MLflow (only if available)
        if use_mlflow:
            print("\nLogging model to MLflow...")
            try:
                mlflow.pytorch.log_model(
                    model,
                    artifact_path="model",
                    registered_model_name=f"{config.get('dataset', 'dataset')}_{model_name}",
                    input_example=dummy_input
                )
            except Exception as e:
                print(f"Warning: Failed to log model to MLflow: {e}")
        
        # Save training summary
        with open(os.path.join(full_save_dir, 'training_summary.txt'), 'w') as f:
            f.write(f"Training Summary\n")
            f.write(f"================\n\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Dataset: {config.get('dataset')}\n")
            f.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n")
            f.write(f"Test Accuracy: {test_acc:.2f}%\n")
            f.write(f"Total Parameters: {total_params:,}\n")
            f.write(f"Trainable Parameters: {trainable_params:,}\n")
            f.write(f"Logging Backend: {'MLflow' if use_mlflow else 'TensorBoard'}\n")
        
        if use_mlflow:
            mlflow.log_artifact(os.path.join(full_save_dir, 'training_summary.txt'))
        
        if writer is not None:
            writer.close()
        
        print("\n✅ Training completed successfully!")
        print(f"Results saved to: {full_save_dir}")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
        print(f"Test accuracy: {test_acc:.2f}%")
        print(f"Logging backend: {'MLflow' if use_mlflow else 'TensorBoard'}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train action recognition models')
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    args = parser.parse_args()
    
    main(args.config)