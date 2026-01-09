import torch
import torch.nn as nn
from transformers import TimesformerForVideoClassification, AutoImageProcessor


class TimeSformerModel(nn.Module):
    """
    TimeSformer model wrapper for action recognition fine-tuning.
    """
    
    def __init__(self, num_classes=7, pretrained_model="facebook/timesformer-base-finetuned-k400",
                 freeze_backbone=False, dropout=0.3, **kwargs):
        """
        Args:
            num_classes: Number of action classes
            pretrained_model: HuggingFace model identifier
            freeze_backbone: Whether to freeze the backbone during initial training
            dropout: Dropout rate for classifier
        """
        super(TimeSformerModel, self).__init__()
        
        # Load pretrained TimeSformer
        self.model = TimesformerForVideoClassification.from_pretrained(
            pretrained_model,
            num_labels=num_classes,
            ignore_mismatched_sizes=True
        )
        
        # Optionally freeze backbone
        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if 'classifier' not in name:
                    param.requires_grad = False
        
        # Add dropout to classifier if needed
        if dropout > 0:
            self.model.classifier = nn.Sequential(
                nn.Dropout(dropout),
                self.model.classifier
            )
        
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


def get_timesformer_model(num_classes=7, pretrained_model="facebook/timesformer-base-finetuned-k400",
                          freeze_backbone=False, dropout=0.3, **kwargs):
    """
    Factory function to create TimeSformer model.
    
    Args:
        num_classes: Number of action classes
        pretrained_model: HuggingFace model identifier
        freeze_backbone: Whether to freeze backbone
        dropout: Dropout rate
        
    Returns:
        TimeSformerModel instance
    """
    return TimeSformerModel(
        num_classes=num_classes,
        pretrained_model=pretrained_model,
        freeze_backbone=freeze_backbone,
        dropout=dropout
    )