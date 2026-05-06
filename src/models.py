import torch
from torch import nn


class MLPModel(nn.Module):
    def __init__(self, input_size=15, output_size=10):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, output_size),
        )

    def forward(self, x):
        return self.network(x)


class RNNModel(nn.Module):
    def __init__(self, input_size=6, hidden_size=32, output_size=10):
        super().__init__()

        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, hidden = self.rnn(x)
        last_hidden = output[:, -1, :]
        return self.output_layer(last_hidden)


class LSTMModel(nn.Module):
    def __init__(self, input_size=6, hidden_size=32, output_size=10):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.output_layer = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output, hidden = self.lstm(x)
        last_hidden = output[:, -1, :]
        return self.output_layer(last_hidden)