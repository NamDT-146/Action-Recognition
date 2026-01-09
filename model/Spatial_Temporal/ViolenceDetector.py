import torch
import torch.nn as nn
import torch.nn.functional as F
from .CNNExtractor import CNNExtractor
from .LSTM import TemporalModel


class ViolenceDetectionModel(nn.Module):
    """
    Complete Violence Detection Model combining CNN and LSTM
    Following the architecture from BuildModel_basic.py
    """
    
    def __init__(self, 
                 seq_len=20,
                 img_size=244,
                 cnn_arch='resnet50',
                 pretrained=True,
                 freeze_cnn=True,
                 pretrained_coco=False,
                 temporal_model='convlstm',
                 num_layers=3,
                 hidden_dim=256,
                 bidirectional=False,
                 kernel_size=3,
                 num_classes=1,
                 dropout=0.0):
        """
        Args:
            seq_len (int): Sequence length (number of frames)
            img_size (int): Input image size
            cnn_arch (str): CNN architecture ('resnet50', 'vgg19', 'resnet18')
            pretrained (bool): Use ImageNet pretrained weights
            freeze_cnn (bool): Freeze CNN weights during training
            temporal_model (str): Temporal model type ('convlstm', 'lstm')
            num_layers (int): Number of layers in temporal model
            hidden_dim (int): Hidden dimension for temporal model
            kernel_size (int): Kernel size for ConvLSTM
            num_classes (int): Number of output classes
            dropout (float): Dropout rate
        """
        super(ViolenceDetectionModel, self).__init__()
        
        self.seq_len = seq_len
        self.img_size = img_size
        self.num_classes = num_classes
        self.temporal_model_type = temporal_model
        
        # CNN Feature Extractor
        self.cnn_extractor = CNNExtractor(
            cnn_arch=cnn_arch,
            pretrained=pretrained,
            freeze_cnn=freeze_cnn,
            pretrained_coco=pretrained_coco
        )
        
        # Get CNN output dimensions
        if cnn_arch.lower() == 'resnet50':
            cnn_output_dim = 2048
        elif cnn_arch.lower() == 'vgg19':
            cnn_output_dim = 512
        elif cnn_arch.lower() == 'resnet18':
            cnn_output_dim = 512
        elif cnn_arch.lower() == 'efficientnet_b0':
            cnn_output_dim = 1280
        else:
            cnn_output_dim = 2048
        
        # Temporal Model
        self.temporal_model = TemporalModel(
            model_type=temporal_model,
            input_dim=cnn_output_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            return_sequences=False,
            bidirectional=bidirectional
        )
        
        # Classification Head (following BuildModel_basic.py)
        if temporal_model == 'convlstm':
            # For ConvLSTM: MaxPool + Flatten + Dense layers
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.flatten = nn.Flatten()
            
            # Calculate flattened size after maxpool
            # Assuming 7x7 feature maps after CNN, 3x3 after maxpool
            flattened_size = hidden_dim * 3 * 3
            
            self.classifier = nn.Sequential(
                nn.BatchNorm1d(flattened_size),
                nn.Dropout(dropout),
                nn.Linear(flattened_size, 512),
                nn.LeakyReLU(0.16),
                nn.Linear(512, 256),
                nn.Dropout(dropout),
                nn.LeakyReLU(0.16),
                nn.Linear(256, 10),
                nn.Dropout(dropout),
                nn.LeakyReLU(0.16)
            )
        else:
            # For LSTM: Direct dense layers
            self.classifier = nn.Sequential(
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.Linear(256, 10),
                nn.Dropout(dropout),
                nn.ReLU()
            )
        
        # Final prediction layer
        if num_classes == 1:
            self.output_layer = nn.Linear(10, 1)
            self.activation = nn.Sigmoid()
        else:
            self.output_layer = nn.Linear(10, num_classes)
            self.activation = nn.Softmax(dim=1)
    
    def forward(self, x):
        """
        Forward pass
        Args:
            x: Input tensor of shape (batch_size, seq_len, 3, height, width)
        Returns:
            predictions: Output predictions
        """
        # Extract CNN features
        cnn_features = self.cnn_extractor(x)
        
        # Temporal modeling
        if self.temporal_model_type == 'convlstm':
            # ConvLSTM expects (batch_size, seq_len, channels, height, width)
            temporal_features = self.temporal_model(cnn_features)
            
            # Apply maxpool and flatten
            temporal_features = self.maxpool(temporal_features)
            temporal_features = self.flatten(temporal_features)
        else:
            # LSTM expects (batch_size, seq_len, features)
            batch_size, seq_len, c, h, w = cnn_features.size()
            # Global average pooling to flatten spatial dimensions
            cnn_features = F.adaptive_avg_pool2d(cnn_features.view(-1, c, h, w), (1, 1))
            cnn_features = cnn_features.view(batch_size, seq_len, c)
            
            temporal_features = self.temporal_model(cnn_features)
        
        # Classification
        features = self.classifier(temporal_features)
        predictions = self.output_layer(features)
        predictions = self.activation(predictions)
        
        return predictions
    
    def unfreeze_cnn(self):
        """Unfreeze CNN for fine-tuning"""
        self.cnn_extractor.unfreeze_cnn()
    
    def freeze_cnn(self, depth=-1):
        """
        Freeze CNN layers up to a specified depth
        
        Args:
            depth (int): How deep to freeze the CNN layers
                        -1 = freeze all layers (default)
                        0 = don't freeze any layers
                        1 = freeze only the first layer/block
                        2 = freeze the first two layers/blocks
                        ... and so on
        """
        self.cnn_extractor.freeze_cnn_layers(depth)


def build_violence_detection_model(seq_len=20, 
                                 img_size=244,
                                 cnn_arch='efficientnet_b0',
                                 pretrained=True,
                                 freeze_cnn=True,
                                 pretrained_coco=False,
                                 temporal_model='convlstm',
                                 num_layers=3,
                                 hidden_dim=256,
                                 kernel_size=3,
                                 num_classes=1,
                                 dropout=0.0,
                                 learning_rate=1e-4,
                                 optimizer_type='adam',
                                 weight_init='xavier_uniform',
                                 bidirectional=False,
                                 **kwargs):
    """
    Build complete violence detection model
    
    Args:
        seq_len (int): Sequence length
        img_size (int): Input image size
        cnn_arch (str): CNN architecture
        pretrained (bool): Use pretrained weights
        freeze_cnn (bool): Freeze CNN layers
        pretrained_coco (bool): Use COCO pretrained weights
        temporal_model (str): Temporal model type
        num_layers (int): Number of layers in temporal model
        hidden_dim (int): Hidden dimension
        kernel_size (int): ConvLSTM kernel size
        num_classes (int): Number of classes
        dropout (float): Dropout rate
        learning_rate (float): Learning rate
        optimizer_type (str): Optimizer type ('adam', 'rmsprop')
        weight_init (str): Weight initialization method 
                          ('glorot_uniform', 'glorot_normal', 'he_uniform', 'he_normal', 'xavier_uniform', 'xavier_normal')
        bidirectional (bool): Use bidirectional LSTM
        **kwargs: Additional arguments to ignore
    
    Returns:
        tuple: (model, optimizer, criterion)
    """
    
    def init_weights(m):
        """Initialize weights based on layer type and initialization method"""
        if isinstance(m, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            if weight_init == 'glorot_uniform' or weight_init == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight)
            elif weight_init == 'glorot_normal' or weight_init == 'xavier_normal':
                nn.init.xavier_normal_(m.weight)
            elif weight_init == 'he_uniform':
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            elif weight_init == 'he_normal':
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            else:
                # Default to glorot_uniform
                nn.init.xavier_uniform_(m.weight)
            
            # Initialize bias to zero if it exists
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            # BatchNorm layers
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
            
        elif isinstance(m, (nn.LSTM, nn.GRU)):
            # RNN layers
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    if weight_init in ['glorot_uniform', 'xavier_uniform']:
                        nn.init.xavier_uniform_(param)
                    elif weight_init in ['glorot_normal', 'xavier_normal']:
                        nn.init.xavier_normal_(param)
                    elif weight_init == 'he_uniform':
                        nn.init.kaiming_uniform_(param, nonlinearity='relu')
                    elif weight_init == 'he_normal':
                        nn.init.kaiming_normal_(param, nonlinearity='relu')
                    else:
                        nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.constant_(param, 0)
    
    # Create model
    model = ViolenceDetectionModel(
        seq_len=seq_len,
        img_size=img_size,
        cnn_arch=cnn_arch,
        pretrained=pretrained,
        freeze_cnn=freeze_cnn,
        pretrained_coco=pretrained_coco,
        temporal_model=temporal_model,
        num_layers=num_layers,
        hidden_dim=hidden_dim,
        kernel_size=kernel_size,
        num_classes=num_classes,
        dropout=dropout,
        bidirectional=bidirectional
    )
    
    # Apply weight initialization to non-pretrained layers
    if not pretrained or not freeze_cnn:
        # Only initialize layers that are not from pretrained CNN
        for name, module in model.named_modules():
            # Skip CNN extractor if using pretrained weights
            if 'cnn_extractor' in name and pretrained:
                continue
            init_weights(module)
    else:
        # Initialize only the classifier and temporal model layers
        model.temporal_model.apply(init_weights)
        model.classifier.apply(init_weights)
        model.output_layer.apply(init_weights)
        if hasattr(model, 'maxpool'):
            # MaxPool doesn't have weights, so skip
            pass
    
    # Setup optimizer
    if optimizer_type.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type.lower() == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")
    
    # Setup loss function
    if num_classes == 1:
        criterion = nn.BCELoss()
    else:
        criterion = nn.BCEWithLogitsLoss()  # Use logits for multi-class classification
    
    # Print initialization info
    print(f"Model initialized with {weight_init} weight initialization")
    if pretrained and freeze_cnn:
        print("CNN layers use pretrained weights (not reinitialized)")
        print("Only temporal model and classifier layers are initialized")
    else:
        print("All non-pretrained layers are initialized")
    
    return model, optimizer, criterion


# Example usage
if __name__ == "__main__":
    # Test the model
    model, optimizer, criterion = build_violence_detection_model(
        seq_len=20,
        img_size=244,
        cnn_arch='resnet50',
        temporal_model='convlstm',
        num_layers=3,
        hidden_dim=256,
        num_classes=1,
        dropout=0.0
    )
    
    # Test input
    batch_size = 2
    test_input = torch.randn(batch_size, 20, 3, 244, 244)
    
    # Forward pass
    with torch.no_grad():
        output = model(test_input)
        print(f"Model output shape: {output.shape}")
        print(f"Model output: {output}")
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")