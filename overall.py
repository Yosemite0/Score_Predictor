import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# Step 1: Set up CUDA device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Memory Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")

# Step 2: Load and preprocess the dataset
df = pd.read_csv('updated_over_by_over_data_set.csv')

# Add season weights
df['season'] = df['season'].astype(str)
df['season_year'] = df['season'].apply(lambda x: int(x.split('/')[0]))
max_year = 2024
df['season_weight'] = np.exp(-0.1 * (max_year - df['season_year']))

# Encode teams using LabelEncoder + Embedding
team_encoder = LabelEncoder()
df['batting_team'] = team_encoder.fit_transform(df['batting_team'])
df['bowling_team'] = team_encoder.fit_transform(df['bowling_team'])

# Numerical features
numerical_features = ['over', 'run_rate', 'req_runrate', 'target_left', 'season_weight']
scaler = StandardScaler()
numerical_data = scaler.fit_transform(df[numerical_features])

# Step 3: Split the dataset into train and test (by match_id)
match_ids = df['match_id'].unique()
train_match_ids, test_match_ids = train_test_split(match_ids, test_size=0.2, random_state=42)
train_df = df[df['match_id'].isin(train_match_ids)].reset_index(drop=True)
test_df = df[df['match_id'].isin(test_match_ids)].reset_index(drop=True)

# Prepare numerical data for train and test sets
train_numerical_data = scaler.transform(train_df[numerical_features])
test_numerical_data = scaler.transform(test_df[numerical_features])

# Create index mappings for train and test DataFrames
train_index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(train_df.index)}
test_index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(test_df.index)}

# Step 4: Prepare sequences for training and testing
class IPLDataset(Dataset):
    def __init__(self, df, numerical_data, index_map):
        self.df = df
        self.numerical_data = numerical_data
        self.index_map = index_map
        self.matches = []
        for match_id in df['match_id'].unique():
            for inning in [1, 2]:
                match_inning = df[(df['match_id'] == match_id) & (df['inning'] == inning)]
                if len(match_inning) > 1:
                    self.matches.append(match_inning)

    def __len__(self):
        return sum(len(match) - 1 for match in self.matches)

    def __getitem__(self, idx):
        cum_idx = 0
        for match in self.matches:
            match_len = len(match) - 1
            if cum_idx + match_len > idx:
                over_idx = idx - cum_idx
                current_over = match.iloc[:over_idx + 1]
                next_over = match.iloc[over_idx + 1]
                break
            cum_idx += match_len

        # Map the original index to the new index in the split DataFrame
        original_idx = current_over.index[-1]
        mapped_idx = self.index_map[original_idx]

        team1 = torch.tensor(current_over['batting_team'].values[-1], dtype=torch.long)
        team2 = torch.tensor(current_over['bowling_team'].values[-1], dtype=torch.long)
        num_data = torch.tensor(self.numerical_data[mapped_idx], dtype=torch.float32)
        runs = torch.tensor(next_over['total_runs'], dtype=torch.float32)
        wickets = torch.tensor(next_over['is_wicket'], dtype=torch.float32)
        return team1, team2, num_data, runs, wickets

# Create train and test datasets
train_dataset = IPLDataset(train_df, train_numerical_data, train_index_map)
test_dataset = IPLDataset(test_df, test_numerical_data, test_index_map)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True)

# Step 5: Define the model architecture
class IPLModel(nn.Module):
    def __init__(self, num_teams, embedding_dim, numerical_dim, hidden_dim):
        super(IPLModel, self).__init__()
        self.team1_embedding = nn.Embedding(num_teams, embedding_dim)
        self.team2_embedding = nn.Embedding(num_teams, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim * 2 + numerical_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_dim, hidden_dim // 2, batch_first=True)
        self.fc2 = nn.Linear(hidden_dim // 2, 8)
        self.runs_output = nn.Linear(8, 1)
        self.wickets_output = nn.Linear(8, 1)

    def forward(self, team1, team2, num_data):
        team1_embed = self.team1_embedding(team1).squeeze(1)
        team2_embed = self.team2_embedding(team2).squeeze(1)
        x = torch.cat((team1_embed, team2_embed, num_data), dim=1)
        x = self.fc1(x)
        x = self.relu(x)
        x = x.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]
        x = self.fc2(lstm_out)
        x = self.relu(x)
        runs = self.runs_output(x).squeeze()
        wickets = self.wickets_output(x).squeeze()
        wickets = self.relu(wickets)
        return runs, wickets

# Step 6: Initialize and train the model
num_teams = len(team_encoder.classes_)
model = IPLModel(num_teams=num_teams, embedding_dim=10, numerical_dim=len(numerical_features), hidden_dim=32)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Lists to store metrics for plotting
train_losses = []
test_losses = []
train_runs_mae = []
test_runs_mae = []
train_wickets_mae = []
test_wickets_mae = []

num_epochs = 10
for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0
    train_r_mae = 0
    train_w_mae = 0
    for team1, team2, num_data, runs, wickets in train_loader:
        team1, team2, num_data = team1.to(device), team2.to(device), num_data.to(device)
        runs, wickets = runs.to(device), wickets.to(device)

        optimizer.zero_grad()
        pred_runs, pred_wickets = model(team1, team2, num_data)
        loss_runs = criterion(pred_runs, runs)
        loss_wickets = criterion(pred_wickets, wickets)
        loss = loss_runs + loss_wickets
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_r_mae += torch.mean(torch.abs(pred_runs - runs)).item()
        train_w_mae += torch.mean(torch.abs(pred_wickets - wickets)).item()

    train_losses.append(train_loss / len(train_loader))
    train_runs_mae.append(train_r_mae / len(train_loader))
    train_wickets_mae.append(train_w_mae / len(train_loader))

    # Testing phase
    model.eval()
    test_loss = 0
    test_r_mae = 0
    test_w_mae = 0
    with torch.no_grad():
        for team1, team2, num_data, runs, wickets in test_loader:
            team1, team2, num_data = team1.to(device), team2.to(device), num_data.to(device)
            runs, wickets = runs.to(device), wickets.to(device)

            pred_runs, pred_wickets = model(team1, team2, num_data)
            loss_runs = criterion(pred_runs, runs)
            loss_wickets = criterion(pred_wickets, wickets)
            loss = loss_runs + loss_wickets
            test_loss += loss.item()
            test_r_mae += torch.mean(torch.abs(pred_runs - runs)).item()
            test_w_mae += torch.mean(torch.abs(pred_wickets - wickets)).item()

    test_losses.append(test_loss / len(test_loader))
    test_runs_mae.append(test_r_mae / len(test_loader))
    test_wickets_mae.append(test_w_mae / len(test_loader))

    # Print epoch results
    print(f"Epoch {epoch+1}/{num_epochs}:")
    print(f"Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}")
    print(f"Train Runs MAE: {train_runs_mae[-1]:.4f}, Test Runs MAE: {test_runs_mae[-1]:.4f}")
    print(f"Train Wickets MAE: {train_wickets_mae[-1]:.4f}, Test Wickets MAE: {test_wickets_mae[-1]:.4f}")
    if device.type == 'cuda':
        print(f"CUDA Memory Allocated: {torch.cuda.memory_allocated(0)/1024**2:.2f} MB")

# Save the model
torch.save(model.state_dict(), 'ipl_over_prediction_model.pth')

# Step 7: Plot the metrics
epochs = range(1, num_epochs + 1)

# Plot 1: Training vs Testing Loss
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_losses, label='Train Loss', marker='o')
plt.plot(epochs, test_losses, label='Test Loss', marker='o')
plt.title('Training and Testing Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.grid(True)
plt.savefig('loss_over_epochs.png')
plt.close()

# Plot 2: Training vs Testing Runs MAE
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_runs_mae, label='Train Runs MAE', marker='o')
plt.plot(epochs, test_runs_mae, label='Test Runs MAE', marker='o')
plt.title('Training and Testing Runs MAE Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('MAE (Runs)')
plt.legend()
plt.grid(True)
plt.savefig('runs_mae_over_epochs.png')
plt.close()

# Plot 3: Training vs Testing Wickets MAE
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_wickets_mae, label='Train Wickets MAE', marker='o')
plt.plot(epochs, test_wickets_mae, label='Test Wickets MAE', marker='o')
plt.title('Training and Testing Wickets MAE Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('MAE (Wickets)')
plt.legend()
plt.grid(True)
plt.savefig('wickets_mae_over_epochs.png')
plt.close()

print("Plots saved as 'loss_over_epochs.png', 'runs_mae_over_epochs.png', and 'wickets_mae_over_epochs.png'")

# Step 8: User input and prediction
model.eval()
valid_teams = list(team_encoder.classes_)
print("\nAvailable teams:", valid_teams)

while True:
    print("\nEnter details for prediction (or type 'exit' to quit):")
    batting_team = input("Batting team: ").strip()
    if batting_team.lower() == 'exit':
        break
    bowling_team = input("Bowling team: ").strip()
    if bowling_team.lower() == 'exit':
        break

    if batting_team not in valid_teams or bowling_team not in valid_teams:
        print("Invalid team name. Please choose from the available teams.")
        continue

    try:
        over_num = int(input("Current over (1-19): "))
        if over_num < 1 or over_num > 19:
            raise ValueError("Over must be between 1 and 19.")
        run_rate = float(input("Current run rate (e.g., 6.0): "))
        if run_rate < 0:
            raise ValueError("Run rate cannot be negative.")
        req_runrate = float(input("Required run rate (0 if first inning): "))
        if req_runrate < 0:
            raise ValueError("Required run rate cannot be negative.")
        target_left = int(input("Target runs left (0 if first inning): "))
        if target_left < 0:
            raise ValueError("Target left cannot be negative.")
    except ValueError as e:
        print(f"Invalid input: {e}")
        continue

    with torch.no_grad():
        team1 = torch.tensor([team_encoder.transform([batting_team])[0]], dtype=torch.long).to(device)
        team2 = torch.tensor([team_encoder.transform([bowling_team])[0]], dtype=torch.long).to(device)
        num_data = torch.tensor(scaler.transform([[over_num, run_rate, req_runrate, target_left, 1]]), dtype=torch.float32).to(device)
        pred_runs, pred_wickets = model(team1, team2, num_data)
        print(f"\nPrediction for over {over_num + 1}:")
        print(f"Predicted runs: {pred_runs.item():.2f}")
        print(f"Predicted wickets: {pred_wickets.item():.2f} (rounded: {int(round(pred_wickets.item()))})")

# Clear CUDA memory
if device.type == 'cuda':
    torch.cuda.empty_cache()
    print("CUDA memory cleared.")