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
                              B: batch size
                              C: channels (3 for RGB)
                              T: temporal frames
                              H: height
                              W: width
        
        Returns:
            torch.Tensor: Output logits of shape (B, num_classes)
        """
        return self.model(x)
    
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
    
    def freeze_backbone(self, freeze=True):
        """
        Freeze/unfreeze all layers except the final classifier (legacy method).
        
        Args:
            freeze (bool): Whether to freeze the backbone
        """
        num_layers = 5 if freeze else 0
        self.freeze_layers(num_layers_to_freeze=num_layers)


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
    # Example usage
    x3d_model = get_model_x3d(
        model_variant='x3d_s',
        num_classes=10,
        pretrained=True,
        dropout=0.3,
        num_layers_to_freeze=3  # Freeze first 3 blocks, train blocks 3-5
    )
    print(x3d_model)

    # test forward pass with dummy data
    dummy_input = torch.randn(2, 3, 16, 256, 256)  # (B, C, T, H, W)
    output = x3d_model(dummy_input)
    print(f"Output shape: {output.shape}")  # Expected: (2, 10)