from flask import Flask, jsonify
import FinanceDataReader as fdr
import torch
import torch.nn as nn
from datetime import datetime, timedelta, date
import numpy as np
import json


def load_scaler():
    with open('model/model_info.json', 'r') as f:
        model_info = json.load(f)
    f.close()
    scaler = model_info["scaler"]
    return scaler


def minmax_preprocessing(x_data):
    with open('data/minmax.json', 'r') as f:
        minmax_info = json.load(f)
    f.close()
    x_pred = x_data.copy()
    for feature_name in x_data.columns:
        x_pred[feature_name] = (x_data[feature_name] - minmax_info["{}_min".format(feature_name)]) / \
                               (minmax_info["{}_max".format(feature_name)] - minmax_info["{}_min".format(feature_name)])
    x_pred = [x_pred.to_numpy()]
    x_pred = torch.from_numpy(np.array(x_pred))
    return x_pred


def std_preprocessing(x_data):
    with open('data/std.json', 'r') as f:
        std_info = json.load(f)
    f.close()
    x_pred = x_data.copy()
    for feature_name in x_data.columns:
        x_pred[feature_name] = (x_data[feature_name] - std_info["{}_average".format(feature_name)]) / \
                               (std_info["{}_std".format(feature_name)])
    x_pred = [x_pred.to_numpy()]
    x_pred = torch.from_numpy(np.array(x_pred))
    return x_pred


def load_data():
    yesterday = datetime.today() - timedelta(1)
    ks11 = fdr.DataReader('KS11', yesterday - timedelta(45), yesterday)
    ks11 = ks11.iloc[-32:]
    return ks11


def load_model():
    device = torch.device('cpu')
    with open('model/model_info.json', 'r') as f:
        model_info = json.load(f)
    f.close()
    checkpoint = torch.load("model/model.pt", map_location=device)
    if model_info["model"] == "lstm":
        model = LSTM(input_dim=6, hidden_dim=6, output_dim=1, num_layers=model_info["stack"])
    elif model_info["model"] == "gru":
        model = GRU(input_dim=6, hidden_dim=6, output_dim=1, num_layers=model_info["stack"])
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def minmax_postprocessing(pred):
    with open('data/minmax.json', 'r') as f:
        minmax_info = json.load(f)
    f.close()
    pred = pred[0][0]
    pred = pred * (minmax_info["Low_max"] - minmax_info["Low_min"]) + minmax_info["Low_min"]
    return pred


def std_postprocessing(pred):
    with open('data/std.json', 'r') as f:
        std_info = json.load(f)
    f.close()
    pred = pred[0][0]
    pred = pred * std_info["Low_std"] + std_info["Low_average"]
    return pred


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out


class GRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(GRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=x.device).requires_grad_()
        out, (hn) = self.gru(x, (h0.detach()))
        out = self.fc(out[:, -1, :])
        return out


app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def main():
    x_pred = load_data()
    scaler = load_scaler()
    if scaler == "minmax":
        x_pred = minmax_preprocessing(x_pred)
        model = load_model()
        model.eval()
        with torch.no_grad():
            y_pred = model(x_pred.float()).numpy()
        y_pred = minmax_postprocessing(y_pred)
    elif scaler == "std":
        x_pred = minmax_preprocessing(x_pred)
        model = load_model()
        model.eval()
        with torch.no_grad():
            y_pred = model(x_pred.float()).numpy()
        y_pred = std_postprocessing(y_pred)
    result = {"date" : date.today(), "result" : y_pred}
    return jsonify(result)


if __name__ == '__main__':
  app.run(host="0.0.0.0", port="5000")
