import torch
import torch.nn as nn

class MILAdapter(nn.Module):
    """Base class for MIL Adapters."""
    def __init__(self, input_dim=432, hidden_dim=32, dropout=0.6):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (Batch, Segments, Feature_Dim)
        Returns:
            scores: Tensor of shape (Batch, Segments) in range [0, 1]
        """
        raise NotImplementedError


class MIL_MLP(MILAdapter):
    """
    Simple Multi-Layer Perceptron applied to each segment independently.
    Does not explicitly model temporal dependencies, relies on bag-level aggregation.
    
    Architecture optimized for small dataset (313 pairs):
    - Fewer parameters to prevent overfitting
    - High dropout for regularization
    """
    def __init__(self, input_dim=432, hidden_dim=32, dropout=0.6):
        super().__init__(input_dim, hidden_dim, dropout)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Args:
            x: (Batch, Segments, input_dim)
        Returns:
            scores: (Batch, Segments)
        """
        # x: (B, S, C) -> Linear applies to last dim -> (B, S, 1)
        out = self.net(x)  # (B, S, 1)
        return out.squeeze(-1)  # (B, S)


class MIL_LSTM(MILAdapter):
    """
    LSTM adapter to capture temporal evolution of anomalies.
    Standard approach for Weakly Supervised Anomaly Detection (Sultani et al.).
    
    Architecture:
    - Single-layer LSTM with dropout
    - Small hidden size to prevent overfitting on small dataset
    """
    def __init__(self, input_dim=432, hidden_dim=32, dropout=0.6):
        super().__init__(input_dim, hidden_dim, dropout)
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            batch_first=True, 
            bidirectional=False,
            dropout=0.0  # No dropout in single-layer LSTM
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: (Batch, Segments, input_dim)
        Returns:
            scores: (Batch, Segments)
        """
        # x: (B, S, C)
        lstm_out, _ = self.lstm(x)  # (B, S, hidden_dim)
        lstm_out = self.dropout(lstm_out)
        out = self.fc(lstm_out)  # (B, S, 1)
        out = self.sigmoid(out)
        return out.squeeze(-1)  # (B, S)


class MIL_Conv1D(MILAdapter):
    """
    1D Convolutional adapter to capture local temporal patterns.
    
    Architecture:
    - Multiple 1D conv layers with different kernel sizes
    - Captures local temporal dependencies
    - Fewer parameters than LSTM
    """
    def __init__(self, input_dim=432, hidden_dim=32, dropout=0.6):
        super().__init__(input_dim, hidden_dim, dropout)
        
        # Project to lower dimension first
        self.proj = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 1D convolutions (input: (B, C, S))
        self.conv1 = nn.Conv1d(256, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(128, hidden_dim, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Final projection
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Args:
            x: (Batch, Segments, input_dim)
        Returns:
            scores: (Batch, Segments)
        """
        # Project features
        x = self.proj(x)  # (B, S, 256)
        
        # Transpose for conv1d: (B, S, C) -> (B, C, S)
        x = x.transpose(1, 2)  # (B, 256, S)
        
        # Conv layers
        x = self.relu(self.conv1(x))  # (B, 128, S)
        x = self.dropout(x)
        x = self.relu(self.conv2(x))  # (B, hidden_dim, S)
        x = self.dropout(x)
        
        # Transpose back: (B, C, S) -> (B, S, C)
        x = x.transpose(1, 2)  # (B, S, hidden_dim)
        
        # Final projection
        out = self.fc(x)  # (B, S, 1)
        out = self.sigmoid(out)
        return out.squeeze(-1)  # (B, S)


def get_adapter(adapter_type: str, input_dim: int = 432, hidden_dim: int = 32, dropout: float = 0.6):
    """
    Factory function to get adapter model.
    
    Args:
        adapter_type: 'mlp', 'lstm', or 'conv1d'
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension size
        dropout: Dropout rate
        
    Returns:
        MILAdapter instance
    """
    adapter_type = adapter_type.lower()
    
    if adapter_type == 'mlp':
        return MIL_MLP(input_dim, hidden_dim, dropout)
    elif adapter_type == 'lstm':
        return MIL_LSTM(input_dim, hidden_dim, dropout)
    elif adapter_type == 'conv1d':
        return MIL_Conv1D(input_dim, hidden_dim, dropout)
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}. Choose from: mlp, lstm, conv1d")


# Test
if __name__ == "__main__":
    print("="*80)
    print("Testing MIL Adapters")
    print("="*80)
    
    # Test input
    batch_size = 4
    num_segments = 5
    feature_dim = 2048
    
    x = torch.randn(batch_size, num_segments, feature_dim)
    print(f"\nInput shape: {x.shape}")
    
    # Test each adapter
    adapters = {
        'MLP': MIL_MLP(input_dim=feature_dim, hidden_dim=32, dropout=0.6),
        'LSTM': MIL_LSTM(input_dim=feature_dim, hidden_dim=32, dropout=0.6),
        'Conv1D': MIL_Conv1D(input_dim=feature_dim, hidden_dim=32, dropout=0.6)
    }
    
    for name, adapter in adapters.items():
        print(f"\n{'='*60}")
        print(f"Testing {name} Adapter")
        print(f"{'='*60}")
        
        # Count parameters
        num_params = sum(p.numel() for p in adapter.parameters())
        print(f"Parameters: {num_params:,}")
        
        # Forward pass
        with torch.no_grad():
            scores = adapter(x)
        
        print(f"Output shape: {scores.shape}")
        print(f"Output range: [{scores.min():.4f}, {scores.max():.4f}]")
        print(f"Sample scores: {scores[0]}")
    
    print("\n" + "="*80)
    print("All adapter tests passed!")
    print("="*80)