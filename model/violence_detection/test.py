from train import test_model
import torch

def test_from_checkpoint(checkpoint_path):
    """
    Test the model on the test set using only the checkpoint path.
    All config and model info are loaded from the checkpoint and dataset code.
    """
    # Load config and build model, criterion, and test_loader
    from model import build_violence_detection_model
    from optflow2d_dataset import create_data_loaders

    # Infer device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # You may need to adjust these defaults to match your training config
    config = {
        'seq_length': 16,
        'img_size': 256,
        'cnn_arch': 'efficientnet_b0',
        'pretrained': True,
        'freeze_cnn': False,
        'pretrained_coco': False,
        'temporal_model': 'convlstm',
        'hidden_dim': 256,
        'num_classes': 2,
        'bidirectional': True,
        'dropout': 0.3,
        'learning_rate': 1e-4,
        'optimizer_type': 'adam',
        'weight_init': 'xavier_uniform',
        'batch_size': 4,
        'num_workers': 8,
        'channels': 3,
        'figure_size': 256,
    }

    # Build model and criterion
    model, optimizer, criterion = build_violence_detection_model(
        seq_len=config['seq_length'],
        img_size=config['img_size'],
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

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Create test loader
    split_file = 'data/precomputed/split_info.csv'
    _, _, test_loader = create_data_loaders(
        split_file=split_file,
        batch_size=config['batch_size'],
        num_frames=config['seq_length'],
        num_workers=config['num_workers'],
        mode='rgb',
    )

    # Run test
    print(f"Testing model from checkpoint: {checkpoint_path}")
    test_loss, test_metrics = test_model(model, test_loader, criterion, device)
    print("Test Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall: {test_metrics['recall']:.4f}")
    print(f"Test F1-Score: {test_metrics['f1_score']:.4f}")

# Example usage:
if __name__ == '__main__':
    # Replace with your actual checkpoint path
    test_from_checkpoint('weights/rgb_lstm_cnn_8323.pth')
# test_from_checkpoint('checkpoints/best_model_hockeyfight_efficientnet_b0_convlstm_bs16_lr1e-04_20240730_123456.pth')