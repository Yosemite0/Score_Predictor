import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch.nn import Module, LSTM, Linear, Dropout, ReLU, MSELoss, BCEWithLogitsLoss
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, 
                           accuracy_score, precision_score, recall_score, f1_score, 
                           confusion_matrix, roc_curve, auc)
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] -> %(message)s')
logger = logging.getLogger('IPLScoreEvaluator')

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")
if device.type == 'cuda':
    logger.info(f"CUDA Device: {torch.cuda.get_device_name(0)}")


# Define model classes used in training
class IPLMatchDataset(Dataset):
    """PyTorch Dataset for IPL match data with dynamic run rate."""
    
    def __init__(self, df, numerical_data, index_map, target_cols=None, sequence=True):
        self.df = df
        self.numerical_data = numerical_data
        self.index_map = index_map
        self.sequence = sequence
        
        # Default target columns if none specified
        if target_cols is None:
            self.target_cols = ['total_runs', 'is_wicket']
        else:
            self.target_cols = target_cols
            
        logger.info(f"Initializing dataset with {len(df)} records, target columns: {self.target_cols}")
        
        # Group by match_id and inning to build sequences
        if self.sequence:
            self.match_innings = list(df.groupby(['match_id', 'inning']))
            logger.info(f"Created {len(self.match_innings)} match-inning sequences")
        else:
            self.match_innings = None
    
    def __len__(self):
        """Return the number of items in the dataset."""
        if self.sequence:
            return len(self.match_innings)
        return len(self.df)
    
    def __getitem__(self, idx):
        """Get a data item."""
        if self.sequence:
            # Get all overs for this match-inning
            _, group = self.match_innings[idx]
            group = group.sort_values('over')  # Ensure correct order
            
            # Get features
            features = []
            for _, row in group.iterrows():
                # Map original index to the index in numerical_data
                mapped_idx = self.index_map[row.name]
                numerical_features = torch.tensor(self.numerical_data[mapped_idx], dtype=torch.float32)
                
                # Add team encodings
                batting_team = torch.tensor([row['batting_team']], dtype=torch.float32)
                bowling_team = torch.tensor([row['bowling_team']], dtype=torch.float32)
                
                # Combine features
                combined = torch.cat([numerical_features, batting_team, bowling_team])
                features.append(combined)
            
            # Stack features to create sequence
            features = torch.stack(features)
            
            # Get targets for the sequence
            targets = torch.tensor([group[col].values[-1] for col in self.target_cols], dtype=torch.float32)
            sequence_length = torch.tensor(len(group), dtype=torch.int32)
            
            return features, targets, sequence_length
        else:
            # Single over mode
            row = self.df.iloc[idx]
            mapped_idx = self.index_map[row.name]
            
            # Get features
            numerical_features = torch.tensor(self.numerical_data[mapped_idx], dtype=torch.float32)
            batting_team = torch.tensor([row['batting_team']], dtype=torch.float32)
            bowling_team = torch.tensor([row['bowling_team']], dtype=torch.float32)
            features = torch.cat([numerical_features, batting_team, bowling_team])
            
            # Get targets
            targets = torch.tensor([row[col] for col in self.target_cols], dtype=torch.float32)
            
            return features, targets


class IPLScorePredictor(Module):
    """Neural network model for predicting IPL match scores with dynamic run rate."""
    
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3, output_size=2):
        super(IPLScorePredictor, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layers
        self.lstm = LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected layers for prediction
        self.fc1 = Linear(hidden_size, hidden_size // 2)
        self.dropout = Dropout(dropout)
        self.relu = ReLU()
        self.fc2 = Linear(hidden_size // 2, output_size)
        
        logger.info(
            f"Initialized model: input_size={input_size}, "
            f"hidden_size={hidden_size}, num_layers={num_layers}, "
            f"dropout={dropout}, output_size={output_size}"
        )
    
    def forward(self, x, seq_lengths=None):
        """Forward pass through the network."""
        batch_size = x.size(0)
        
        # Pack padded sequence if sequence lengths are provided
        if seq_lengths is not None:
            # Pack the sequence to handle variable length sequences efficiently
            x = pack_padded_sequence(x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.data.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.data.device)
        
        # Forward through LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Unpack if we packed earlier
        if seq_lengths is not None:
            out, _ = pad_packed_sequence(out, batch_first=True)
            
            # Use the last valid output for each sequence based on its length
            # Convert seq_lengths to indices tensor of correct type
            indices = (seq_lengths - 1).view(-1, 1).expand(batch_size, self.hidden_size)
            indices = indices.unsqueeze(1).to(torch.int64).to(x.data.device)
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


def calculate_weighted_run_rate(dataframe, alpha=0.7):
    """Calculate weighted run rate based on previous overs with an exponential weighting."""
    logger.info(f"Calculating weighted run rate with alpha={alpha}")
    
    # Create a copy of the dataframe to avoid modifying the original
    df = dataframe.copy()
    
    # Calculate per-over run rate
    df['curr_run_rate'] = df['total_runs']
    
    # Initialize column for weighted run rate
    df['weighted_run_rate'] = 0.0
    
    # Process each match and inning separately
    match_innings_groups = df.groupby(['match_id', 'inning'])
    logger.info(f"Processing {len(match_innings_groups)} match-inning combinations")
    
    for (match_id, inning), group in match_innings_groups:
        # Sort by over to ensure correct sequence
        group = group.sort_values('over')
        
        # Initialize weighted run rate with the first over's run rate
        weighted_rr = group['curr_run_rate'].iloc[0]
        
        # Process each row and update the weighted run rate
        for idx, row in group.iterrows():
            if row['over'] == 1:  # First over - just use its run rate
                df.at[idx, 'weighted_run_rate'] = row['curr_run_rate']
            else:
                # Apply the formula: α(RR_current) + (1-α)(RR_previous_weighted)
                weighted_rr = alpha * row['curr_run_rate'] + (1-alpha) * weighted_rr
                df.at[idx, 'weighted_run_rate'] = weighted_rr
    
    # Round for readability
    df['weighted_run_rate'] = df['weighted_run_rate'].round(2)
    
    # Sort the dataframe for readability
    df = df.sort_values(['match_id', 'inning', 'over'])
    
    logger.info("Weighted run rate calculation completed")
    return df


def process_and_prepare_data(input_path='cleaned_data/updated_over_by_over_data_set.csv', 
                           output_path='cleaned_data/weighted_run_rate_dataset.csv',
                           alpha=0.9,
                           test_size=0.2,
                           random_state=42):
    """Main function to process data and prepare it for modeling."""
    # Load the dataset
    logger.info(f"Loading data from {input_path}")
    try:
        df = pd.read_csv(input_path)
        logger.info(f"Loaded data with shape: {df.shape}")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None
    
    # Calculate weighted run rate
    df = calculate_weighted_run_rate(df, alpha=alpha)
    
    # Save the processed data
    logger.info(f"Saving processed data to {output_path}")
    df.to_csv(output_path, index=False)
    logger.info(f"Data saved successfully. Shape: {df.shape}")
    
    # Process season information
    df['season'] = df['season'].astype(str)
    df['season_year'] = df['season'].apply(lambda x: int(x.split('/')[0]))
    
    # Encode teams
    team_encoder = LabelEncoder()
    df['batting_team'] = team_encoder.fit_transform(df['batting_team'])
    df['bowling_team'] = team_encoder.transform(df['bowling_team'])
    logger.info(f"Encoded {len(team_encoder.classes_)} unique teams")
    
    # Define and scale numerical features
    numerical_features = ['over', 'curr_run_rate', 'req_runrate', 'target_left', 'weighted_run_rate']
    scaler = StandardScaler()
    df_numerical = pd.DataFrame(
        scaler.fit_transform(df[numerical_features]), 
        columns=numerical_features
    )
    logger.info(f"Scaled {len(numerical_features)} numerical features")
    
    # Split by match_id to keep all overs of a match together
    match_ids = df['match_id'].unique()
    train_match_ids, test_match_ids = train_test_split(
        match_ids, test_size=test_size, random_state=random_state
    )
    
    train_df = df[df['match_id'].isin(train_match_ids)].reset_index(drop=True)
    test_df = df[df['match_id'].isin(test_match_ids)].reset_index(drop=True)
    
    # Create index mappings for dataset creation
    train_index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(train_df.index)}
    test_index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(test_df.index)}
    
    logger.info(f"Training set: {len(train_df)} rows, {len(train_match_ids)} matches")
    logger.info(f"Testing set: {len(test_df)} rows, {len(test_match_ids)} matches")
    
    # Prepare numerical data for train and test sets
    train_numerical_data = scaler.transform(train_df[numerical_features])
    test_numerical_data = scaler.transform(test_df[numerical_features])
    
    # Display sample data for verification
    print("\nSample data (first 5 rows):")
    display_cols = ['match_id', 'inning', 'over', 'total_runs', 'weighted_run_rate']
    print(df[display_cols].head())
    
    return {
        'processed_df': df,
        'train_df': train_df,
        'test_df': test_df,
        'train_numerical_data': train_numerical_data,
        'test_numerical_data': test_numerical_data,
        'train_index_map': train_index_map,
        'test_index_map': test_index_map,
        'team_encoder': team_encoder,
        'scaler': scaler,
        'numerical_features': numerical_features
    }


def evaluate_model(model, test_loader, sequence_mode=True, save_dir='results'):
    """Evaluate the model on test data and visualize predictions vs actual values."""
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Enable evaluation mode
    model.eval()
    model.to(device)
    
    # Initialize lists to store predictions and actual values
    all_run_preds = []
    all_run_actual = []
    all_wicket_preds = []
    all_wicket_actual = []
    
    # Initialize metrics
    run_mae_total = 0
    wicket_correct = 0
    total_samples = 0
    
    logger.info("Starting model evaluation on test data")
    
    with torch.no_grad():
        for batch in test_loader:
            # Process batch based on sequence mode
            if sequence_mode:
                x_batch, y_batch, seq_lengths = batch
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                seq_lengths = seq_lengths.to(device)
                outputs = model(x_batch, seq_lengths)
            else:
                x_batch, y_batch = batch
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(x_batch)
            
            # Extract predictions
            run_preds = outputs[:, 0].cpu().numpy()  # Runs are direct values
            wicket_probs = torch.sigmoid(outputs[:, 1]).cpu().numpy()  # Wicket probabilities
            wicket_preds = (wicket_probs > 0.5).astype(float)  # Binary predictions (0 or 1)
            
            # Extract actual values
            run_actual = y_batch[:, 0].cpu().numpy()
            wicket_actual = y_batch[:, 1].cpu().numpy()
            
            # Store for visualization
            all_run_preds.extend(run_preds)
            all_run_actual.extend(run_actual)
            all_wicket_preds.extend(wicket_preds)
            all_wicket_actual.extend(wicket_actual)
            
            # Calculate metrics
            run_mae = np.mean(np.abs(run_preds - run_actual))
            run_mae_total += run_mae * len(run_preds)
            wicket_correct += np.sum(wicket_preds == wicket_actual)
            total_samples += len(wicket_actual)
    
    # Calculate overall metrics
    overall_run_mae = run_mae_total / total_samples
    overall_wicket_accuracy = wicket_correct / total_samples
    
    logger.info(f"Evaluation completed - Run MAE: {overall_run_mae:.2f}, Wicket Accuracy: {overall_wicket_accuracy:.2%}")
    
    # Convert lists to arrays for easier handling
    all_run_preds = np.array(all_run_preds)
    all_run_actual = np.array(all_run_actual)
    all_wicket_preds = np.array(all_wicket_preds)
    all_wicket_actual = np.array(all_wicket_actual)
    
    # Visualize run predictions
    plt.figure(figsize=(10, 6))
    plt.scatter(all_run_actual, all_run_preds, alpha=0.5)
    plt.plot([min(all_run_actual), max(all_run_actual)], [min(all_run_actual), max(all_run_actual)], 'r--')
    plt.xlabel('Actual Runs')
    plt.ylabel('Predicted Runs')
    plt.title('Actual vs Predicted Runs')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'actual_vs_predicted_runs.png'))
    plt.close()
    
    # Visualize wicket predictions (ROC curve)
    # Get predicted probabilities for wickets
    with torch.no_grad():
        all_wicket_probs = []
        for batch in test_loader:
            if sequence_mode:
                x_batch, _, seq_lengths = batch
                x_batch = x_batch.to(device)
                seq_lengths = seq_lengths.to(device)
                outputs = model(x_batch, seq_lengths)
            else:
                x_batch, _ = batch
                x_batch = x_batch.to(device)
                outputs = model(x_batch)
                
            wicket_probs = torch.sigmoid(outputs[:, 1]).cpu().numpy()
            all_wicket_probs.extend(wicket_probs)
    
    # Calculate ROC curve
    all_wicket_probs = np.array(all_wicket_probs)
    # Ensure binary values for ROC calculation
    binary_wicket_actual = (all_wicket_actual > 0).astype(int)
    fpr, tpr, _ = roc_curve(binary_wicket_actual, all_wicket_probs)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) for Wicket Prediction')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(save_dir, 'wicket_prediction_roc.png'))
    plt.close()
    
    # Show comparison of actual vs predicted runs and wickets
    # Select a random sample for visualization
    sample_size = min(50, len(all_run_actual))
    indices = np.random.choice(len(all_run_actual), sample_size, replace=False)
    
    # Create indices for x-axis
    x_indices = np.arange(sample_size)
    
    # Plot runs comparison
    plt.figure(figsize=(12, 6))
    plt.plot(x_indices, all_run_actual[indices], 'b-', label='Actual Runs')
    plt.plot(x_indices, all_run_preds[indices], 'r--', label='Predicted Runs')
    plt.xlabel('Sample Index')
    plt.ylabel('Runs')
    plt.title('Comparison of Actual vs Predicted Runs')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'actual_vs_predicted_comparison.png'))
    plt.close()
    
    # Calculate additional metrics
    run_metrics = {
        'MAE': mean_absolute_error(all_run_actual, all_run_preds),
        'MSE': mean_squared_error(all_run_actual, all_run_preds),
        'RMSE': np.sqrt(mean_squared_error(all_run_actual, all_run_preds)),
        'R2': r2_score(all_run_actual, all_run_preds)
    }
    
    # Convert wicket data to binary for metric calculation
    binary_wicket_actual = (all_wicket_actual > 0).astype(int)
    binary_wicket_preds = (all_wicket_preds > 0).astype(int)
    
    wicket_metrics = {
        'Accuracy': accuracy_score(binary_wicket_actual, binary_wicket_preds),
        'Precision': precision_score(binary_wicket_actual, binary_wicket_preds, zero_division=0),
        'Recall': recall_score(binary_wicket_actual, binary_wicket_preds, zero_division=0),
        'F1': f1_score(binary_wicket_actual, binary_wicket_preds, zero_division=0),
        'AUC': roc_auc
    }
    
    # Display confusion matrix for wickets
    cm = confusion_matrix(binary_wicket_actual, binary_wicket_preds)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix for Wicket Prediction')
    plt.colorbar()
    plt.xticks([0, 1], ['No Wicket', 'Wicket'])
    plt.yticks([0, 1], ['No Wicket', 'Wicket'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    
    # Add text annotations to confusion matrix
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'wicket_confusion_matrix.png'))
    plt.close()
    
    # Print metrics
    print("\nRun Prediction Metrics:")
    for metric, value in run_metrics.items():
        print(f"{metric}: {value:.4f}")
        
    print("\nWicket Prediction Metrics:")
    for metric, value in wicket_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return {
        'run_metrics': run_metrics,
        'wicket_metrics': wicket_metrics,
        'predictions': {
            'run_preds': all_run_preds,
            'run_actual': all_run_actual,
            'wicket_preds': all_wicket_preds,
            'wicket_actual': all_wicket_actual,
            'wicket_probs': all_wicket_probs
        }
    }


def main():
    """Run model evaluation on test data."""
    logger.info("Starting model evaluation process")
    
    # Load and prepare data
    logger.info("Loading and preparing data")
    input_path = 'cleaned_data/updated_over_by_over_data_set.csv'
    output_path = 'cleaned_data/weighted_run_rate_dataset.csv'
    
    data = process_and_prepare_data(
        input_path=input_path, 
        output_path=output_path,
        alpha=0.7
    )
    
    if data is None:
        logger.error("Failed to load or process data")
        return
    
    # Create test dataset
    test_dataset = IPLMatchDataset(
        data['test_df'],
        data['test_numerical_data'],
        data['test_index_map'],
        sequence=True
    )
    
    # Collate function for sequences of varying lengths
    def collate_fn(batch):
        # Separate features, targets, and sequence lengths
        features = [item[0] for item in batch]
        targets = [item[1] for item in batch]
        seq_lengths = [item[2] for item in batch]
        
        # Pad sequences
        features_padded = pad_sequence(features, batch_first=True)
        targets_batch = torch.stack(targets)
        seq_lengths_batch = torch.stack(seq_lengths)
        
        return features_padded, targets_batch, seq_lengths_batch
    
    # Create DataLoader for test data
    batch_size = 32
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Calculate input size
    sample_batch = next(iter(test_loader))
    input_size = sample_batch[0].shape[-1]
    
    # Create model with same parameters as training
    logger.info("Creating model instance")
    model_params = {
        'input_size': input_size,
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.3,
        'output_size': 2  # Runs and wickets
    }
    
    model = IPLScorePredictor(**model_params)
    
    # Load model weights
    logger.info("Loading model weights")
    try:
        checkpoint = torch.load('best_ipl_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        logger.info(f"Model loaded successfully. Last validation loss: {checkpoint.get('val_loss', 'N/A')}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    # Run evaluation
    logger.info("Running model evaluation")
    evaluation_results = evaluate_model(
        model=model,
        test_loader=test_loader,
        sequence_mode=True,
        save_dir='results'
    )
    
    # Display summary
    logger.info("Evaluation completed")
    logger.info(f"Run Prediction MAE: {evaluation_results['run_metrics']['MAE']:.2f} runs")
    logger.info(f"Run Prediction R2 Score: {evaluation_results['run_metrics']['R2']:.4f}")
    logger.info(f"Wicket Prediction Accuracy: {evaluation_results['wicket_metrics']['Accuracy']:.2%}")
    logger.info(f"Wicket Prediction F1 Score: {evaluation_results['wicket_metrics']['F1']:.4f}")
    
    # Additional detailed results are saved in the results directory
    logger.info("Plots and results saved in the results directory")


if __name__ == "__main__":
    main()