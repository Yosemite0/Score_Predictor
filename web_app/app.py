from flask import Flask, render_template, request
from model import IPLModel
import os, json

app = Flask(__name__)


with open('config.json', 'r') as f:
    config = json.load(f)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/predict', methods=['POST'])
def predict():
    #dummy prediction logic
    data = request.json
    prediction = "dummy prediction"
    IPLModel = IPLModel(model_path_lstm=config['MODEL_PATH_LSTM'], model_path_lr=config['MODEL_PATH_LR'])
    team1 = data['team1']
    team2 = data['team2']
    over_data = data['over_data']
    target = data.get('target', None)
    prediction_lstm = IPLModel.predict_lstm(team1, team2, over_data, target)
    prediction_lr = IPLModel.predict_lr(team1, team2, over_data, target)
    prediction = {
        "lstm": prediction_lstm,
        "lr": prediction_lr
    }
    return {"prediction": prediction}

