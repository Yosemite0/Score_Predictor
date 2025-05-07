import torch
import torch.nn as nn
import logging

logger = logging.getLogger('IPLScorePredictor')

class IPLScorePredictor(nn.Module):
    """Neural network model for predicting IPL match scores.
    
    This model uses LSTM layers to process sequence data and predict
    final scores and wickets.
    """
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3, output_size=2):
        """Initialize the model.
        
        Args:
            input_size: Size of input features
            hidden_size: Size of LSTM hidden layers
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            output_size: Size of output (typically 2 for runs and wickets)
        """
        super(IPLScorePredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers for prediction
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size // 2, output_size)
        
        logger.info(
            f"Initialized model: input_size={input_size}, "
            f"hidden_size={hidden_size}, num_layers={num_layers}, "
            f"dropout={dropout}, output_size={output_size}"
        )
    
    def forward(self, x, seq_lengths=None):
        """Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            seq_lengths: Actual sequence lengths for each item in the batch
            
        Returns:
            Predicted scores with runs as direct values and wickets as logits
        """
        batch_size = x.size(0)
        
        # Pack padded sequence if sequence lengths are provided
        if seq_lengths is not None:
            # Pack the sequence to handle variable length sequences efficiently
            from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
            x = pack_padded_sequence(x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.data.device if hasattr(x, 'data') else x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.data.device if hasattr(x, 'data') else x.device)
        
        # Forward through LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Unpack if we packed earlier
        if seq_lengths is not None:
            out, _ = pad_packed_sequence(out, batch_first=True)
            
            # Use the last valid output for each sequence based on its length
            # Convert seq_lengths to indices tensor of correct type
            indices = (seq_lengths - 1).view(-1, 1).expand(batch_size, self.hidden_size)
            indices = indices.unsqueeze(1).to(torch.int64).to(out.device) # Ensure indices are on the same device as 'out'
            out = out.gather(1, indices).squeeze(1)
        else:
            # Use the output from the last time step if no sequence lengths
            out = out[:, -1, :]
        
        # Forward through fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)  # Output contains both runs prediction and wicket logits
        
        return out

class CombinedIPLLoss(nn.Module):
    """Custom loss function for IPL score prediction that combines MSE for runs and BCE for wickets.
    
    This loss function handles the multi-task nature of the problem by applying appropriate
    loss functions to each component of the prediction.
    """
    def __init__(self, runs_weight=1.0, wicket_weight=1.0):
        """Initialize the combined loss function.
        
        Args:
            runs_weight: Weight for the runs prediction loss
            wicket_weight: Weight for the wicket prediction loss
        """
        super(CombinedIPLLoss, self).__init__()
        self.runs_weight = runs_weight
        self.wicket_weight = wicket_weight
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()  # Includes sigmoid activation
        logger.info(f"Initialized Combined IPL Loss with weights - Runs: {runs_weight}, Wickets: {wicket_weight}")
    
    def forward(self, y_pred, y_true):
        """Calculate the combined loss.
        
        Args:
            y_pred: Model predictions with shape (batch_size, 2)
                   where [:, 0] are runs predictions and [:, 1] are wicket logits
            y_true: Ground truth values with shape (batch_size, 2)
                   where [:, 0] are actual runs and [:, 1] are binary wicket indicators
            
        Returns:
            Combined weighted loss
        """
        # Extract run and wicket components
        run_pred = y_pred[:, 0]  # Runs are direct predictions
        wicket_pred = y_pred[:, 1]  # Wicket predictions as logits
        
        run_true = y_true[:, 0]  # Actual runs
        wicket_true = y_true[:, 1]  # Actual wickets (0 or 1)
        
        # Calculate individual losses
        run_loss = self.mse_loss(run_pred, run_true)
        wicket_loss = self.bce_loss(wicket_pred, wicket_true)
        
        # Combine losses with weights
        combined_loss = self.runs_weight * run_loss + self.wicket_weight * wicket_loss
        
        return combined_loss, run_loss, wicket_loss