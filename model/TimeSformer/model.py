import torch
import torch.nn as nn
from transformers import TimesformerForVideoClassification, AutoImageProcessor


class TimeSformerModel(nn.Module):
    """
    TimeSformer model wrapper for action recognition fine-tuning.
    """
    
    def __init__(self, num_classes=7, pretrained_model="facebook/timesformer-base-finetuned-k400",
                 freeze_backbone=False, freeze_layers=10, dropout=0.3, **kwargs):
        """
        Args:
            num_classes: Number of action classes
            pretrained_model: HuggingFace model identifier
            freeze_backbone: Whether to freeze the backbone during initial training
            freeze_layers: Number of encoder layers to freeze (0-12). Default: 10
                          Only applies if freeze_backbone=True
            dropout: Dropout rate for classifier
        """
        super(TimeSformerModel, self).__init__()
        
        # Load pretrained TimeSformer
        self.model = TimesformerForVideoClassification.from_pretrained(
            pretrained_model,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
        # Optionally freeze backbone with layer-wise control
        if freeze_backbone:
            self._freeze_layers(freeze_layers)
        
        # Add dropout to classifier if needed
        if dropout > 0:
            self.model.classifier = nn.Sequential(
                nn.Dropout(dropout),
                self.model.classifier
            )
    
    def _freeze_layers(self, num_layers_to_freeze):
        """
        Freeze specific number of encoder layers.
        
        Args:
            num_layers_to_freeze: Number of layers to freeze (0-12)
        """
        # Clamp to valid range
        num_layers_to_freeze = max(0, min(12, num_layers_to_freeze))
        
        # Freeze patch embeddings and positional embeddings
        for param in self.model.timesformer.embeddings.parameters():
            param.requires_grad = False
        
        # Freeze specified encoder layers
        for i in range(num_layers_to_freeze):
            for param in self.model.timesformer.encoder.layer[i].parameters():
                param.requires_grad = False
        
        print(f"Frozen first {num_layers_to_freeze} encoder layers out of 12")
        print(f"Trainable layers: {12 - num_layers_to_freeze} encoder layers + classifier")
        
    def forward(self, pixel_values):
        """
        Forward pass.
        
        Args:
            pixel_values: Tensor of shape (batch_size, num_frames, channels, height, width)
            
        Returns:
            Logits of shape (batch_size, num_classes)
        """
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits
    
    def unfreeze_backbone(self):
        """Unfreeze all parameters for fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True
        print("All layers unfrozen")


def get_timesformer_model(num_classes=7, pretrained_model="facebook/timesformer-base-finetuned-k400",
                          freeze_backbone=False, freeze_layers=10, dropout=0.3, **kwargs):
    """
    Factory function to create TimeSformer model.
    
    Args:
        num_classes: Number of action classes
        pretrained_model: HuggingFace model identifier
        freeze_backbone: Whether to freeze backbone
        freeze_layers: Number of encoder layers to freeze (0-12). Default: 10
        dropout: Dropout rate
        
    Returns:
        TimeSformerModel instance
    """
    return TimeSformerModel(
        num_classes=num_classes,
        pretrained_model=pretrained_model,
        freeze_backbone=freeze_backbone,
        freeze_layers=freeze_layers,
        dropout=dropout
    )