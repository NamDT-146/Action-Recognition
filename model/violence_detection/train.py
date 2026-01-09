import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.utils.tensorboard import SummaryWriter
import os
import time
import logging
from datetime import datetime
import numpy as np
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

from model import ViolenceDetectionModel, build_violence_detection_model
# from dataset import VideoDataset, create_data_loaders
# from optflow2ddataset import create_data_loaders
# from precompute_optflowdataset import create_data_loaders
from optflow2d_dataset import create_data_loaders

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve"""
    def __init__(self, patience=300, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

def save_checkpoint(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }, filepath)
    logger.info(f"Checkpoint saved: {filepath}")

def load_checkpoint(model, optimizer, filepath):
    """Load model checkpoint"""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    accuracy = checkpoint['accuracy']
    logger.info(f"Checkpoint loaded: {filepath}")    
    return epoch, loss, accuracy

def calculate_metrics(y_true, y_pred, y_pred_proba=None):
    """Calculate various metrics"""
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    return metrics

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer=None):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    batch_losses = []
    
    for batch_idx, (sequences, labels) in enumerate(train_loader):
        # print(sequences.shape, labels.shape)  # Debugging line to check shapes        
        sequences = sequences.to(device).float()
        # labels = labels.to(device).float() # for one hot encoding
        # labels = labels.to(device).float().view(-1, 1)
        labels = labels.to(device).long().view(-1)
        labels = torch.nn.functional.one_hot(labels, num_classes=2).float()
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(sequences)
        # print (outputs)  # Debugging line to check outputs
        # print(outputs, labels)  # Debugging line to check shapes
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        batch_loss = loss.item()
        running_loss += batch_loss
        batch_losses.append(batch_loss)
        
        # Store predictions and labels for metrics
        predictions = (outputs > 0.5).cpu().numpy().astype(int)
        all_predictions.extend(predictions.flatten())
        all_labels.extend(labels.cpu().numpy().astype(int).flatten())
        
        # Log to tensorboard every 10 batches
        if writer and batch_idx % 10 == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Train/Batch_Loss', batch_loss, global_step)
            
            # Log gradients
            for name, param in model.named_parameters():
                if param.grad is not None:
                    writer.add_histogram(f'Gradients/{name}', param.grad, global_step)
                    writer.add_scalar(f'Gradient_Norm/{name}', param.grad.norm().item(), global_step)
        
        if batch_idx % 10 == 0:
            logger.info(f'Train Epoch: {epoch} [{batch_idx * len(sequences)}/{len(train_loader.dataset)} '
                       f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    epoch_loss = running_loss / len(train_loader)
    metrics = calculate_metrics(all_labels, all_predictions)
    
    return epoch_loss, metrics, batch_losses

def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for sequences, labels in val_loader:
            sequences = sequences.to(device).float()
            # labels = labels.to(device).float() # for one hot encoding
            # labels = labels.to(device).float().view(-1, 1)
            labels = labels.to(device).long().view(-1)
            labels = torch.nn.functional.one_hot(labels, num_classes=2).float()
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            # Store predictions and labels for metrics
            predictions = (outputs > 0.5).cpu().numpy().astype(int)
            all_predictions.extend(predictions.flatten())
            all_labels.extend(labels.cpu().numpy().astype(int).flatten())
    
    epoch_loss = running_loss / len(val_loader)
    metrics = calculate_metrics(all_labels, all_predictions)
    
    return epoch_loss, metrics

def test_model(model, test_loader, criterion, device, writer=None, epoch=None):
    """Test the model"""
    model.eval()
    running_loss = 0.0
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences = sequences.to(device).float()
            # labels = labels.to(device).float() # for one hot encoding
            # labels = labels.to(device).float().view(-1, 1)
            labels = labels.to(device).long().view(-1)
            labels = torch.nn.functional.one_hot(labels, num_classes=2).float()            
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            # Store predictions, labels, and probabilities
            probabilities = outputs.cpu().numpy()
            predictions = (outputs > 0.5).cpu().numpy().astype(int)
            
            all_predictions.extend(predictions.flatten())
            all_labels.extend(labels.cpu().numpy().astype(int).flatten())
            all_probabilities.extend(probabilities.flatten())
    
    test_loss = running_loss / len(test_loader)
    metrics = calculate_metrics(all_labels, all_predictions, all_probabilities)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    logger.info(f"Confusion Matrix:\n{cm}")
    
    # Log test results to tensorboard
    if writer and epoch is not None:
        writer.add_scalar('Test/Loss', test_loss, epoch)
        writer.add_scalar('Test/Accuracy', metrics['accuracy'], epoch)
        writer.add_scalar('Test/Precision', metrics['precision'], epoch)
        writer.add_scalar('Test/Recall', metrics['recall'], epoch)
        writer.add_scalar('Test/F1_Score', metrics['f1_score'], epoch)
        
        # Log confusion matrix as heatmap
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=['Non-Violence', 'Violence'],
               yticklabels=['Non-Violence', 'Violence'],
               title='Confusion Matrix',
               ylabel='True label',
               xlabel='Predicted label')
        
        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                       ha="center", va="center",
                       color="white" if cm[i, j] > thresh else "black")
        
        writer.add_figure('Test/Confusion_Matrix', fig, epoch)
        plt.close(fig)
    
    return test_loss, metrics

def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies, save_path):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    # Accuracy plot
    ax2.plot(train_accuracies, label='Train Accuracy')
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def log_model_architecture(writer, model, input_shape):
    """Log model architecture to tensorboard"""
    try:
        # Create dummy input
        dummy_input = torch.randn(1, *input_shape)
        writer.add_graph(model.cpu(), dummy_input)
        model.cuda()  # Move back to GPU if available
        logger.info("Model architecture logged to TensorBoard")
    except Exception as e:
        logger.warning(f"Could not log model architecture to TensorBoard: {e}")

def print_model_architecture(model, input_shape):
    """Print detailed model architecture"""
    print("\n" + "="*80)
    print("MODEL ARCHITECTURE")
    print("="*80)
    
    # Print model summary
    print(f"Model: {model.__class__.__name__}")
    print(f"Input shape: {input_shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"\nParameter Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Frozen parameters: {frozen_params:,}")
    print(f"  Trainable percentage: {100 * trainable_params / total_params:.2f}%")
    
    # Print layer-by-layer details
    print(f"\nLayer Details:")
    print("-" * 80)
    print(f"{'Layer Name':<30} {'Type':<20} {'Parameters':<15} {'Trainable':<10}")
    print("-" * 80)
    
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:  # Only leaf modules
            param_count = sum(p.numel() for p in module.parameters())
            trainable = any(p.requires_grad for p in module.parameters())
            module_type = module.__class__.__name__
            
            if param_count > 0:
                print(f"{name:<30} {module_type:<20} {param_count:<15,} {'Yes' if trainable else 'No':<10}")
    
    print("-" * 80)
    
    # Print model structure
    print(f"\nModel Structure:")
    print("-" * 50)
    print(model)
    print("="*80)

def main(device=None, resume=False, checkpoint_dir=None):
    # Training configuration
    config = {
        'data_dir': "data",
        'dataset_name': "hockeyfight",  # Change to your dataset name
        'batch_size': 2,
        'figure_size': 224,
        'channels': 3,  # RGB images
        'seq_length': 16,
        'crop_dark': (11, 38),
        'num_workers': 2,
        'epochs': 300,
        'learning_rate': 1e-4,
        'optimizer_type': 'adam',  # 'adam' or 'rmsprop'
        'weight_init': 'xavier_uniform',  # Weight initialization method
        'weight_decay': 1e-5,
        'patience_es': 15,  # Early stopping patience
        'patience_lr': 3,   # Learning rate reduction patience
        'lr_factor': 0.5,   # Learning rate reduction factor
        'dropout': 0.3,  # Dropout rate
        'cnn_arch': 'efficientnet_b0',  # 'resnet50', 'vgg19', 'resnet18', 'efficientnet_b0'
        'temporal_model': 'convlstm',
        'hidden_dim': 128,
        'bidirectional': False,
        'num_layers': 1,
        'pretrained': True,
        'freeze_cnn': False,
        'pretrained_coco': False,
        'checkpoint_dir': 'checkpoints',
        'results_dir': 'results',
        'tensorboard_dir': 'runs',
        'device': device,  # Add device to config
        'num_classes': 2,  # Binary classification
    }
    
    # Create directories
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    os.makedirs(config['results_dir'], exist_ok=True)
    os.makedirs(config['tensorboard_dir'], exist_ok=True)
    
    # Create unique experiment name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{config['dataset_name']}_{config['cnn_arch']}_{config['temporal_model']}_bs{config['batch_size']}_lr{config['learning_rate']:.0e}_{timestamp}"
    
    # Initialize TensorBoard writer
    tensorboard_path = os.path.join(config['tensorboard_dir'], experiment_name)
    writer = SummaryWriter(tensorboard_path)
    logger.info(f"TensorBoard logs will be saved to: {tensorboard_path}")
    
    # Log hyperparameters
    writer.add_text('Hyperparameters', str(config))
    
    # Device setup
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    logger.info(f"Using device: {device}")
    writer.add_text('Device', str(device))
    
    # Create DataLoaders
    logger.info("Creating data loaders...")
    split_file = 'data/precomputed_ensemble/split_info.csv'
    train_loader, val_loader, test_loader = create_data_loaders(
        split_file=split_file,
        batch_size=config['batch_size'],
        num_frames=config['seq_length'],
        num_workers=config['num_workers'],
        mode='flow', 
        flow_mag_threshold=0.2, 
        as_flow_rgb=True
        # one_hot=False
    )

    # small_train_dataset = torch.utils.data.Subset(train_loader.dataset, range(128))  # Use only 128 samples
    # train_loader = torch.utils.data.DataLoader(
    #     small_train_dataset,
    #     batch_size=config['batch_size'],
    #     shuffle=True,
    #     num_workers=config['num_workers'],
    #     pin_memory=True
    # )
            
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Log dataset info to tensorboard
    writer.add_scalar('Dataset/Train_Samples', len(train_loader.dataset), 0)
    writer.add_scalar('Dataset/Val_Samples', len(val_loader.dataset), 0)
    writer.add_scalar('Dataset/Test_Samples', len(test_loader.dataset), 0)
    
    # Create model
    logger.info("Creating model...")
    model, optimizer, criterion = build_violence_detection_model(
        seq_len=config['seq_length'],
        img_size=config['figure_size'],
        cnn_arch=config['cnn_arch'],
        pretrained=config['pretrained'],
        freeze_cnn=config['freeze_cnn'],
        pretrained_coco=config['pretrained_coco'],
        temporal_model=config['temporal_model'],
        hidden_dim=config['hidden_dim'],
        num_classes=config['num_classes'],
        bidirectional=config['bidirectional'],
        dropout=config['dropout'],
        learning_rate=config['learning_rate'],
        optimizer_type=config['optimizer_type'],
        weight_init=config['weight_init']
    )
    
    # Add weight decay to optimizer
    if config['optimizer_type'].lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), 
                                   lr=config['learning_rate'], 
                                   weight_decay=config['weight_decay'])
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), 
                                      lr=config['learning_rate'], 
                                      weight_decay=config['weight_decay'])
        
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Log model info to tensorboard
    writer.add_scalar('Model/Total_Parameters', total_params, 0)
    writer.add_scalar('Model/Trainable_Parameters', trainable_params, 0)
    
    # Log model architecture (if possible)
    input_shape = (config['seq_length'], config['channels'], config['figure_size'], config['figure_size'])
    log_model_architecture(writer, model, input_shape)

    print_model_architecture(model, input_shape)

    
    model = model.to(device)
    model.freeze_cnn(7)  
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config['lr_factor'], 
                                patience=config['patience_lr'], min_lr=1e-8)
    
    # Learning rate scheduler (StepLR example)
    # scheduler = torch.optim.lr_scheduler.StepLR(
    #     optimizer,
    #     step_size=10,      # Decrease LR every 10 epochs
    #     gamma=0.5          # LR is multiplied by 0.5
    # )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=config['patience_es'])
    
    
    # Training history
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # Training loop
    logger.info("Starting training...")
    start_time = time.time()
    start_epoch = 1
    best_val_accuracy = 0.0
    best_model_path = os.path.join(config['checkpoint_dir'], f'best_model_{experiment_name}.pth')
    if resume and os.path.exists(checkpoint_dir):
        best_model_path = checkpoint_dir
        logger.info(f"Resuming training from checkpoint: {best_model_path}")
        start_epoch, _, best_val_accuracy = load_checkpoint(model, optimizer, best_model_path)
        start_epoch += 1  # Continue from next epoch

    for epoch in range(start_epoch, config['epochs'] + 1):
        epoch_start_time = time.time()
        
        # Train
        train_loss, train_metrics, batch_losses = train_epoch(model, train_loader, criterion, optimizer, device, epoch, writer)
        
        # Validate
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device)
        # val_loss, val_metrics = 0, {'precision': 0, 'recall': 0, 'f1_score': 0, 'accuracy': 0}  # Placeholder
        
        # Update learning rate
        # old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_loss)
        # new_lr = optimizer.param_groups[0]['lr']
        # scheduler.step()
        
        # Store history
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_metrics['accuracy'])
        val_accuracies.append(val_metrics['accuracy'])
        
        epoch_time = time.time() - epoch_start_time
        
        # Log to tensorboard
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Train/Accuracy', train_metrics['accuracy'], epoch)
        writer.add_scalar('Train/Precision', train_metrics['precision'], epoch)
        writer.add_scalar('Train/Recall', train_metrics['recall'], epoch)
        writer.add_scalar('Train/F1_Score', train_metrics['f1_score'], epoch)
        
        writer.add_scalar('Validation/Loss', val_loss, epoch)
        writer.add_scalar('Validation/Accuracy', val_metrics['accuracy'], epoch)
        writer.add_scalar('Validation/Precision', val_metrics['precision'], epoch)
        writer.add_scalar('Validation/Recall', val_metrics['recall'], epoch)
        writer.add_scalar('Validation/F1_Score', val_metrics['f1_score'], epoch)

        writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], epoch)
        writer.add_scalar('Epoch_Time', epoch_time, epoch)
        
        # Log batch losses distribution
        writer.add_histogram('Train/Batch_Losses', np.array(batch_losses), epoch)
        
        # Log model weights and biases
        for name, param in model.named_parameters():
            if param.requires_grad:
                writer.add_histogram(f'Weights/{name}', param, epoch)
                writer.add_scalar(f'Weight_Norm/{name}', param.norm().item(), epoch)
        
        logger.info(f"Epoch {epoch}/{config['epochs']} - {epoch_time:.2f}s")
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_metrics['accuracy']:.4f}")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_metrics['accuracy']:.4f}")
        logger.info(f"Learning Rate: {scheduler.get_last_lr()[0]:.2e}")

        # Save best model
        if val_metrics['accuracy'] > best_val_accuracy:
            best_val_accuracy = val_metrics['accuracy']
            best_model_path = os.path.join(config['checkpoint_dir'], f'best_model_{experiment_name}.pth')
            save_checkpoint(model, optimizer, epoch, val_loss, val_metrics['accuracy'], best_model_path)
            writer.add_scalar('Best_Validation_Accuracy', best_val_accuracy, epoch)
        
        # # Save checkpoint every 10 epochs
        # if epoch % 10 == 0:
        #     checkpoint_path = os.path.join(config['checkpoint_dir'], f'checkpoint_epoch_{epoch}_{experiment_name}.pth')
        #     save_checkpoint(model, optimizer, epoch, val_loss, val_metrics['accuracy'], checkpoint_path)
        
        # Early stopping
        if early_stopping(val_loss, model):
            logger.info(f"Early stopping triggered after epoch {epoch}")
            writer.add_text('Early_Stopping', f'Triggered at epoch {epoch}')
            break
    
    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time/3600:.2f} hours")
    writer.add_scalar('Total_Training_Time_Hours', total_time/3600, epoch)
    
    # Load best model for testing
    best_model_path = os.path.join(config['checkpoint_dir'], f'best_model_{experiment_name}.pth')
    if os.path.exists(best_model_path):
        load_checkpoint(model, optimizer, best_model_path)
    
    # Test the model
    logger.info("Testing model...")
    test_loss, test_metrics = test_model(model, test_loader, criterion, device, writer, epoch)
    
    logger.info("Test Results:")
    logger.info(f"Test Loss: {test_loss:.4f}")
    logger.info(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"Test Precision: {test_metrics['precision']:.4f}")
    logger.info(f"Test Recall: {test_metrics['recall']:.4f}")
    logger.info(f"Test F1-Score: {test_metrics['f1_score']:.4f}")
    
    # Plot training history
    plot_path = os.path.join(config['results_dir'], f'training_history_{experiment_name}.png')
    plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies, plot_path)
    
    # Add training history plot to tensorboard
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.set_title('Model Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    
    ax2.plot(train_accuracies, label='Train Accuracy')
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.set_title('Model Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    writer.add_figure('Training_History', fig, epoch)
    plt.close(fig)
    
    # Save final results
    results = {
        'experiment_name': experiment_name,
        'config': config,
        'best_val_accuracy': best_val_accuracy,
        'test_metrics': test_metrics,
        'total_training_time': total_time,
        'total_epochs': len(train_losses),
        'tensorboard_path': tensorboard_path
    }
    
    import json
    results_path = os.path.join(config['results_dir'], f'training_results_{experiment_name}.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")
    logger.info(f"TensorBoard logs saved to {tensorboard_path}")
    logger.info(f"To view TensorBoard: tensorboard --logdir {config['tensorboard_dir']}")
    
    # Close tensorboard writer
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Violence Detection Model')
    parser.add_argument('--device', type=str, default=None, 
                       help='Device to use for training (e.g., "cuda", "cuda:0", "cpu"). If not specified, will use CUDA if available, otherwise CPU.')
    parser.add_argument('--resume', action='store_true', help='Resume training from best model checkpoint')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='Directory to save model checkpoints')
    args = parser.parse_args()
    main(device=args.device, resume=args.resume, checkpoint_dir=args.checkpoint_dir)