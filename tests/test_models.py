import torch

from src.models import MLPModel, RNNModel, LSTMModel


def test_mlp_output_shape():
    model = MLPModel()
    x = torch.zeros((2, 15))

    y = model(x)

    assert y.shape == (2, 10)


def test_rnn_output_shape():
    model = RNNModel()
    x = torch.zeros((2, 10, 6))

    y = model(x)

    assert y.shape == (2, 10)


def test_lstm_output_shape():
    model = LSTMModel()
    x = torch.zeros((2, 10, 6))

    y = model(x)

    assert y.shape == (2, 10)