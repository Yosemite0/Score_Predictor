import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader

# Custom Dataset
class IPLDataset(Dataset):
    def __init__(self, sequences, target_runss):
        """
        Initialize the IPLDataset.
        
        Args:
            sequences: Tensor of shape (batch_size, max_overs, feature_dim) containing match sequences
            target_runss: Tensor of shape (batch_size, 3) containing [total_runs, total_overs, total_wickets]
        """
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.target_runss = torch.tensor(target_runss, dtype=torch.float32)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.target_runss[idx]

# Model Definition without Venue Embedding
class IPLModel(nn.Module):
    def __init__(self, num_teams, feature_dim=10, hidden_dim=32, lstm_dim=64, temperature=1.0):
        super(IPLModel, self).__init__()
        self.temperature = temperature
        # Embeddings
        self.team_embedding = nn.Embedding(num_teams, 8)
        
        # Input dimension after embeddings
        embed_dim = 6 + 8 + 8 + 1  # numerical (6) + batting_team (8) + bowling_team (8) + is_batting_first (1)
        
        # Layers
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)  # Dropout after FC
        self.lstm = nn.LSTM(hidden_dim, lstm_dim, batch_first=True)
        self.dropout2 = nn.Dropout(0.3)  # Dropout after LSTM
        self.fc2 = nn.Linear(lstm_dim, 3)  # Output: runs, overs, wickets
    
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        
        # Split features - make sure indices match create_sequences columns order
        # Columns: 'over', 'run', 'wicket', 'run_rate', 'req_runrate', 'target_runs', 'batting_team', 'bowling_team', 'is_batting_first'
        numerical = x[:, :, :6]  # 0-5: over, run, wicket, run_rate, req_runrate, target_runs
        batting_team = x[:, :, 6].long()  # 6: batting_team
        bowling_team = x[:, :, 7].long()  # 7: bowling_team
        is_batting_first = x[:, :, 8:9]  # 8: is_batting_first
        
        # Apply embeddings
        batting_team_emb = self.team_embedding(batting_team)
        bowling_team_emb = self.team_embedding(bowling_team)
        
        # Concatenate
        x = torch.cat([numerical, batting_team_emb, bowling_team_emb, is_batting_first], dim=-1)
        
        # FC Layer
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        # LSTM
        x, _ = self.lstm(x)
        x = self.dropout2(x)
        
        # Apply temperature scaling
        x = x[:, -1, :] / self.temperature
        
        # Final FC Layer
        x = self.fc2(x)
        
        return x

# Load and preprocess data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    
    # Handle missing values
    df['is_batting_first'] = (df['who_is_batting_first'] == df['batting_team']).astype(int)
    df['req_runrate'] = df['req_runrate'].fillna(0)
    df['target_runs'] = df['target_runs'].fillna(0)
    
    # Encode categorical variables
    le_team = LabelEncoder()
    all_teams = pd.concat([df['batting_team'], df['bowling_team']]).unique()
    le_team.fit(all_teams)
    df['batting_team'] = le_team.transform(df['batting_team'])
    df['bowling_team'] = le_team.transform(df['bowling_team'])
    
    # Normalize numerical features
    scaler = StandardScaler()
    numerical_cols = ['over', 'total_runs', 'is_wicket', 'run_rate', 'req_runrate', 'target_runs']
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    return df, le_team, None, scaler

# Create sequences
def create_sequences(df, max_overs=20):
    sequences = []
    target_runss = []
    match_ids = df['match_id'].unique()
    
    for match_id in match_ids:
        match_data = df[df['match_id'] == match_id].sort_values('over')
        
        # Check if the required columns exist and adapt
        if 'total_runs' in match_data.columns:
            run_col = 'total_runs'
        else:
            run_col = 'run'
            
        if 'is_wicket' in match_data.columns:
            wicket_col = 'is_wicket'
        else:
            wicket_col = 'wicket'
            
        features = match_data[['over', run_col, wicket_col, 'run_rate', 'req_runrate', 
                              'target_runs', 'batting_team', 'bowling_team', 
                              'is_batting_first']].values
        
        # Compute actual totals after 20 overs
        total_runs = match_data[run_col].sum()
        total_wickets = match_data[wicket_col].sum()
        total_overs = min(match_data['over'].max(), max_overs)
        
        # Pad sequence
        padded_features = np.zeros((max_overs, features.shape[1]))
        seq_len = min(len(features), max_overs)
        padded_features[:seq_len] = features[:seq_len]
        
        sequences.append(padded_features)
        target_runss.append([total_runs, total_overs, total_wickets])
    
    return np.array(sequences), np.array(target_runss), np.array(match_ids)

# Training function with early stopping
def train_model(model, train_loader, test_loader, device, epochs=50, patience=5):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # L2 regularization
    
    best_test_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                test_loss += loss.item()
        
        train_loss /= len(train_loader)
        test_loss /= len(test_loader)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}')
        
        # Early stopping
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_model_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    return model

# Main execution
def main(file_path):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load and preprocess
    df, le_team, le_venue, scaler = load_and_preprocess_data(file_path)
    
    # Create sequences
    X, y, match_ids = create_sequences(df)
    
    # Sort by match_id and take last 10% for test
    sorted_indices = np.argsort(match_ids)
    X = X[sorted_indices]
    y = y[sorted_indices]
    match_ids = match_ids[sorted_indices]
    
    test_size = 0.1
    n_test = int(len(X) * test_size)
    X_train = X[:-n_test]
    y_train = y[:-n_test]
    X_test = X[-n_test:]
    y_test = y[-n_test:]
    
    # Create datasets and loaders
    train_dataset = IPLDataset(X_train, y_train)
    test_dataset = IPLDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Initialize model
    num_teams = len(le_team.classes_)
    model = IPLModel(num_teams, temperature=1.0).to(device)
    
    # Train with early stopping
    model = train_model(model, train_loader, test_loader, device)
    
    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler_state_dict': scaler,
        'team_encoder': le_team,
        # No venue encoder used in this version
    }, 'best_ipl_model.pth')
    
    return model, scaler, le_team, None

if __name__ == '__main__':
    file_path = './cleaned_data/updated_over_by_over_data_set.csv'  # Replace with your CSV file path
    model, scaler, le_team, le_venue = main(file_path)