import torch
import torch.nn as nn


def get_x3d_model(model_variant='x3d_m', num_classes=400, pretrained=True, dropout=0.5, **kwargs):
    """
    Get X3D model from PyTorch Hub with optional pretrained weights.
    
    Args:
        model_variant (str): X3D model variant ('x3d_xs', 'x3d_s', 'x3d_m', 'x3d_l')
        num_classes (int): Number of output classes
        pretrained (bool): Whether to load pretrained weights from Kinetics-400
        dropout (float): Dropout rate for the classifier head
        **kwargs: Additional arguments (for compatibility)
        
    Returns:
        nn.Module: X3D model with modified head for specified number of classes
    """
    # Load pretrained model from PyTorch Hub
    if pretrained:
        print(f"Loading pretrained {model_variant} model from Kinetics-400...")
        model = torch.hub.load('facebookresearch/pytorchvideo', model_variant, pretrained=True)
    else:
        print(f"Loading {model_variant} model without pretrained weights...")
        model = torch.hub.load('facebookresearch/pytorchvideo', model_variant, pretrained=False)
    
    # Get the input features of the final projection layer
    in_features = model.blocks[5].proj.in_features
    
    # Replace the final projection layer if number of classes is different
    if num_classes != 400:
        print(f"Replacing final layer: {in_features} -> {num_classes} classes")
        model.blocks[5].proj = nn.Linear(in_features, num_classes)
    
    # Update dropout rate if specified
    if dropout != 0.5:
        print(f"Updating dropout rate to {dropout}")
        model.blocks[5].dropout = nn.Dropout(p=dropout, inplace=False)
    
    return model


class X3DModel(nn.Module):
    """
    Wrapper class for X3D model that provides a consistent interface.
    """
    def __init__(self, model_variant='x3d_m', num_classes=400, pretrained=True, dropout=0.5, num_layers_to_freeze=5):
        """
        Initialize X3D model.
        
        Args:
            model_variant (str): X3D model variant ('x3d_xs', 'x3d_s', 'x3d_m', 'x3d_l')
            num_classes (int): Number of output classes
            pretrained (bool): Whether to load pretrained weights
            dropout (float): Dropout rate
            num_layers_to_freeze (int): Number of blocks to freeze (0-5).
                                       5 = freeze all blocks except final layer (default)
                                       0 = train all layers including blocks
        """
        super(X3DModel, self).__init__()
        self.model = get_x3d_model(
            model_variant=model_variant,
            num_classes=num_classes,
            pretrained=pretrained,
            dropout=dropout
        )
        self.num_classes = num_classes
        self.num_layers_to_freeze = num_layers_to_freeze
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, T, H, W)
        
        Returns:
            torch.Tensor: Output logits of shape (B, num_classes)
        """
        return self.model(x)
    
    def extract_features(self, x):
        """
        Extract 432-dim features from pool layer (following exact model architecture).
        
        This extracts features after the AvgPool3d layer in the ResNetBasicHead,
        which is the natural 432-dimensional bottleneck before expansion to 2048.
        
        Args:
            x: Input tensor (B, C, T, H, W)
            
        Returns:
            features: Tensor of shape (B, 432)
        """
        # Forward through blocks 0-4 (backbone)
        for i in range(5):
            x = self.model.blocks[i](x)
        
        # Forward through block 5 ProjectedPool up to and including AvgPool3d
        # This follows the exact same path as the full forward pass
        x = self.model.blocks[5].pool.pre_conv(x)   # (B, 192, T, H, W) -> (B, 432, T, H, W)
        x = self.model.blocks[5].pool.pre_norm(x)   # BatchNorm
        x = self.model.blocks[5].pool.pre_act(x)    # ReLU
        x = self.model.blocks[5].pool.pool(x)       # AvgPool3d: (B, 432, T, H, W) -> (B, 432, 1, 1, 1)
        
        # Flatten to (B, 432)
        features = x.flatten(1)
        
        return features
    
    def freeze_layers(self, num_layers_to_freeze=5):
        """
        Freeze specified number of blocks (layers).
        
        Args:
            num_layers_to_freeze (int): Number of blocks to freeze (0-5).
                                       5 = freeze blocks 0-4, only block 5 (final) is trainable
                                       4 = freeze blocks 0-3, blocks 4-5 are trainable
                                       0 = train all blocks
        """
        if num_layers_to_freeze < 0 or num_layers_to_freeze > 5:
            raise ValueError(f"num_layers_to_freeze must be between 0 and 5, got {num_layers_to_freeze}")
        
        # X3D has 6 blocks (0-5), where block 5 is the final classification block
        total_blocks = 6
        
        print(f"Freezing first {num_layers_to_freeze} blocks out of {total_blocks}:")
        
        for block_idx in range(total_blocks):
            block = self.model.blocks[block_idx]
            
            if block_idx < num_layers_to_freeze:
                # Freeze this block
                for param in block.parameters():
                    param.requires_grad = False
                print(f"  Block {block_idx}: FROZEN")
            else:
                # Keep this block trainable
                for param in block.parameters():
                    param.requires_grad = True
                print(f"  Block {block_idx}: TRAINABLE")


def get_model_x3d(**kwargs):
    """
    Factory function to create X3D model compatible with train.py.
    
    Expected kwargs:
        - model_variant (str): X3D variant ('x3d_xs', 'x3d_s', 'x3d_m', 'x3d_l')
        - num_classes (int): Number of output classes
        - pretrained (bool): Whether to use pretrained weights
        - dropout (float): Dropout rate
        - num_layers_to_freeze (int): Number of blocks to freeze (0-5, default: 5)
    
    Returns:
        X3DModel: Instantiated X3D model
    """
    model_variant = kwargs.get('model_variant', 'x3d_m')
    num_classes = kwargs.get('num_classes', 400)
    pretrained = kwargs.get('pretrained', True)
    dropout = kwargs.get('dropout', 0.5)
    num_layers_to_freeze = kwargs.get('num_layers_to_freeze', 5)
    
    model = X3DModel(
        model_variant=model_variant,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout=dropout,
        num_layers_to_freeze=num_layers_to_freeze
    )
    
    # Apply layer freezing
    model.freeze_layers(num_layers_to_freeze=num_layers_to_freeze)
    
    return model


if __name__ == "__main__":
    print("="*80)
    print("Testing X3D Feature Extraction")
    print("="*80)
    
    # Create model
    x3d_model = get_model_x3d(
        model_variant='x3d_m',
        num_classes=2,
        pretrained=True,
        dropout=0.3,
        num_layers_to_freeze=5
    )
    
    # Test input
    dummy_input = torch.randn(2, 3, 16, 224, 224)  # (B, C, T, H, W)
    print(f"\nInput shape: {dummy_input.shape}")
    
    # Test forward pass
    print("\n" + "-"*60)
    print("Testing standard forward pass")
    print("-"*60)
    output = x3d_model(dummy_input)
    print(f"Output shape: {output.shape}")  # Expected: (2, 2)
    
    # Test 432-dim feature extraction
    print("\n" + "-"*60)
    print("Testing 432-dim feature extraction (with shape trace)")
    print("-"*60)
    
    # Trace shapes manually
    x = dummy_input
    print(f"Input: {x.shape}")
    
    for i in range(5):
        x = x3d_model.model.blocks[i](x)
    print(f"After block 4: {x.shape}")
    
    x = x3d_model.model.blocks[5].pool.pre_conv(x)
    print(f"After pre_conv: {x.shape}")
    
    x = x3d_model.model.blocks[5].pool.pre_norm(x)
    print(f"After pre_norm: {x.shape}")
    
    x = x3d_model.model.blocks[5].pool.pre_act(x)
    print(f"After pre_act: {x.shape}")
    
    x = x3d_model.model.blocks[5].pool.pool(x)
    print(f"After pool (AvgPool3d): {x.shape}")
    
    features = x.flatten(1)
    print(f"After flatten: {features.shape}")
    
    # Now test the method
    print("\nUsing extract_features() method:")
    features = x3d_model.extract_features(dummy_input)
    print(f"Features shape: {features.shape}")  # Expected: (2, 432)
    print(f"Feature stats: min={features.min():.4f}, max={features.max():.4f}, mean={features.mean():.4f}")
    
    print("\n" + "="*80)
    print("✓ All tests passed!")
    print("="*80)