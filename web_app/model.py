import torch as t
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple

class LSTM_Model:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Define the model architecture
        self.model = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size)
        )

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        with t.no_grad():
            x = t.tensor(x, dtype=t.float32)
            return self.model(x).numpy()
    def load_model(self, model_path):
        self.model.load_state_dict(t.load(model_path))
        self.model.eval()

class LR_Model:
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

        # Define the model architecture
        self.model = nn.Linear(self.input_size, self.output_size)

    def forward(self, x):
        return self.model(x)

    def predict(self, x):
        with t.no_grad():
            x = t.tensor(x, dtype=t.float32)
            return self.model(x).numpy()

    def load_model(self, model_path):
        self.model.load_state_dict(t.load(model_path))
        self.model.eval()

class IPLModel:
    def __init__(self, model_path_lstm, model_path_lr):
        self.lstm_model = LSTM_Model(input_size=10, hidden_size=20, output_size=1) 
        self.lstm_model.load_model(model_path_lstm)
        self.lr_model = LR_Model(input_size=10, output_size=1)
        self.lr_model.load_model(model_path_lr)

    def predict_lstm(self, team1, team2, over_data: list, target=None) -> Tuple[float, float]:
        pass
    
    def predict_lr(self, team1, team2, over_data: list, target=None) -> Tuple[float, float]:
        pass


    
