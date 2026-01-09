import os
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet50_Weights, VGG19_Weights, ResNet18_Weights


class CNNExtractor(nn.Module):
    """
    CNN Feature Extractor for video frames
    Supports ResNet50, VGG19, ResNet18 architectures
    """
    
    def __init__(self, cnn_arch='efficientnet_b0', pretrained=True, freeze_cnn=False, pretrained_coco=False):
        """
        Args:
            cnn_arch (str): CNN architecture ('resnet50', 'vgg19', 'resnet18')
            pretrained (bool): Use ImageNet pretrained weights
            freeze_cnn (bool): Freeze CNN weights during training
            coco_pretrained (bool): Use COCO-pretrained ResNet50 backbone (from Faster R-CNN)
        """
        super(CNNExtractor, self).__init__()
        
        self.cnn_arch = cnn_arch.lower()
        self.freeze_cnn = freeze_cnn
        

        if self.cnn_arch == 'efficientnet_b0':
            if pretrained:
                local_weight_path = os.path.join('pretrained_weight', 'efficientnet_b0.pth')
                self.cnn = models.efficientnet_b0(weights=None)
                state_dict = torch.load(local_weight_path, map_location='cpu')
                self.cnn.load_state_dict(state_dict)
            else:
                self.cnn = models.efficientnet_b0(weights=None)
            self.feature_dim = 1280
            self.cnn = self.cnn.features  # Only feature layers
        elif self.cnn_arch == 'resnet50' and pretrained_coco:
            # Use COCO-pretrained backbone from Faster R-CNN
            detection_model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            self.cnn = detection_model.backbone.body
            self.feature_dim = 2048
        elif self.cnn_arch == 'resnet50':
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            self.cnn = models.resnet50(weights=weights)
            self.feature_dim = self.cnn.fc.in_features
            self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])  # Remove avgpool and fc
        elif self.cnn_arch == 'vgg19':
            weights = VGG19_Weights.IMAGENET1K_V1 if pretrained else None
            self.cnn = models.vgg19(weights=weights)
            self.feature_dim = 512
            self.cnn = self.cnn.features  # Only feature layers
        elif self.cnn_arch == 'resnet18':
            weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.cnn = models.resnet18(weights=weights)
            self.feature_dim = self.cnn.fc.in_features
            self.cnn = nn.Sequential(*list(self.cnn.children())[:-2])  # Remove avgpool and fc
        else:
            raise ValueError(f"Unsupported CNN architecture: {cnn_arch}")
        
        # Freeze CNN parameters if specified
        if self.freeze_cnn:
            for param in self.cnn.parameters():
                param.requires_grad = False
                
        # Add adaptive pooling to ensure consistent output size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))
  
        
    def forward(self, x):
        """
        Forward pass through CNN
        Args:
            x: Input tensor of shape (batch_size, seq_len, 3, height, width)
        Returns:
            features: Extracted features of shape (batch_size, seq_len, feature_dim, 7, 7)
        """
        batch_size, seq_len, c, h, w = x.size()
        
        # Reshape to process all frames at once
        x = x.view(batch_size * seq_len, c, h, w)
        
        # Extract features
        features = self.cnn(x)
        
        # Apply adaptive pooling
        features = self.adaptive_pool(features)
        
        # Reshape back to sequence format
        _, feat_c, feat_h, feat_w = features.size()
        features = features.view(batch_size, seq_len, feat_c, feat_h, feat_w)
        
        return features
    
    def get_output_shape(self, input_shape):
        """
        Get output shape for given input shape
        Args:
            input_shape: (seq_len, channels, height, width)
        Returns:
            output_shape: (seq_len, feature_dim, 7, 7)
        """
        if self.cnn_arch == 'efficientnet_b0':
            return (input_shape[0], 1280, 7, 7)
        if self.cnn_arch == 'resnet50':
            return (input_shape[0], 2048, 7, 7)
        elif self.cnn_arch == 'vgg19':
            return (input_shape[0], 512, 7, 7)
        elif self.cnn_arch == 'resnet18':
            return (input_shape[0], 512, 7, 7)
        
    def unfreeze_cnn(self):
        """Unfreeze CNN parameters for fine-tuning"""
        for param in self.cnn.parameters():
            param.requires_grad = True
        self.freeze_cnn = False
        
    # In CNNExtractor.py
    def freeze_cnn_layers(self, depth=-1):
        """
        Freeze CNN layers up to specified depth
        
        Args:
            depth (int): How deep to freeze the CNN layers
                        -1 = freeze all layers (default)
                        0 = don't freeze any layers
                        1 = freeze only the first layer/block
                        2 = freeze the first two layers/blocks
                        ... and so on
        """
        if depth == 0:
            # Don't freeze any layers
            return
        
        # Get list of named modules
        named_modules = list(self.cnn.named_children())
        
        # Determine how many modules to freeze
        if depth == -1:
            # Freeze all layers
            freeze_count = len(named_modules)
        else:
            # Freeze specified number of layers/blocks
            freeze_count = min(depth, len(named_modules))
        
        # Freeze parameters in selected modules
        for i, (name, module) in enumerate(named_modules):
            if i < freeze_count:
                for param in module.parameters():
                    param.requires_grad = False
                print(f"Froze CNN layer/block: {name}")
            else:
                # Ensure remaining layers are trainable
                for param in module.parameters():
                    param.requires_grad = True
                print(f"Keeping CNN layer/block trainable: {name}")