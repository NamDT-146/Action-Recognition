import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    """
    ConvLSTM Cell implementation
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias
        
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )
        
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        
        # Concatenate along channel axis
        combined = torch.cat([input_tensor, h_cur], dim=1)
        
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        
        return h_next, c_next
    
    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    """
    ConvLSTM implementation similar to Keras ConvLSTM2D
    """
    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers=1, 
                 bias=True, return_sequences=False, dropout=0.0):
        super(ConvLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.bias = bias
        self.return_sequences = return_sequences
        self.dropout = nn.Dropout(dropout)
        
        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                        hidden_dim=self.hidden_dim,
                                        kernel_size=self.kernel_size,
                                        bias=self.bias))
        
        self.cell_list = nn.ModuleList(cell_list)
        
    def forward(self, input_tensor, hidden_state=None):
        """
        Args:
            input_tensor: 5D tensor (batch_size, seq_len, input_dim, height, width)
            hidden_state: Initial hidden state
        Returns:
            layer_output_list: List of outputs for each layer
            last_state_list: List of last states for each layer
        """
        batch_size, seq_len, _, height, width = input_tensor.size()
        
        # Initialize hidden state if not provided
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size, (height, width))
        
        layer_output_list = []
        last_state_list = []
        
        cur_layer_input = input_tensor
        
        for layer_idx in range(self.num_layers):
            h, c = hidden_state[layer_idx]
            output_inner = []
            
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](cur_layer_input[:, t, :, :, :], (h, c))
                output_inner.append(h)
            
            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = self.dropout(layer_output)
            
            layer_output_list.append(layer_output)
            last_state_list.append((h, c))
        
        if self.return_sequences:
            return layer_output_list[-1], last_state_list
        else:
            return layer_output_list[-1][:, -1, :, :, :], last_state_list
    
    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states


class LSTMClassifier(nn.Module):
    """
    LSTM-based classifier for temporal modeling
    """
    def __init__(self, input_dim, hidden_dim, num_layers=3, dropout=0.0, 
                 return_sequences=False, bidirectional=False):
        super(LSTMClassifier, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.return_sequences = return_sequences
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
        Returns:
            output: LSTM output
            hidden: Final hidden state
        """
        lstm_out, hidden = self.lstm(x)
        
        if self.return_sequences:
            return self.dropout(lstm_out), hidden
        else:
            # Return only the last output
            return self.dropout(lstm_out[:, -1, :]), hidden
    
    def get_output_dim(self):
        """Get output dimension"""
        return self.hidden_dim * (2 if self.bidirectional else 1)


class TemporalModel(nn.Module):
    """
    Temporal modeling component that can use either ConvLSTM or regular LSTM
    """
    def __init__(self, model_type='convlstm', input_dim=2048, hidden_dim=128, 
                 kernel_size=3, num_layers=1, dropout=0.0, return_sequences=False, bidirectional=False):
        super(TemporalModel, self).__init__()
        
        self.model_type = model_type.lower()
        self.return_sequences = return_sequences
        
        if self.model_type == 'convlstm':
            self.temporal_layer = ConvLSTM(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                kernel_size=kernel_size,
                num_layers=num_layers,
                dropout=dropout,
                return_sequences=return_sequences
            )
        elif self.model_type == 'lstm':
            self.temporal_layer = LSTMClassifier(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                return_sequences=return_sequences,
                bidirectional=bidirectional
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def forward(self, x):
        """
        Args:
            x: Input tensor 
               - For ConvLSTM: (batch_size, seq_len, channels, height, width)
               - For LSTM: (batch_size, seq_len, features)
        Returns:
            Temporal features
        """
        if self.model_type == 'convlstm':
            output, _ = self.temporal_layer(x)
            return output
        else:
            output, _ = self.temporal_layer(x)
            return output