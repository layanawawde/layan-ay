import torch
from torch import nn

from src.main import create_example
from src.models import MLPModel, RNNModel, LSTMModel


def build_mlp_input(example):
    values = (
        example["noisy_samples"]
        + example["frequency_one_hot"]
        + [example["sigma"]]
    )
    return torch.tensor(values, dtype=torch.float32)


def build_sequence_input(example):
    sequence = []

    for sample in example["noisy_samples"]:
        row = [sample] + example["frequency_one_hot"] + [example["sigma"]]
        sequence.append(row)

    return torch.tensor(sequence, dtype=torch.float32)


def build_dataset(size=200):
    mlp_inputs = []
    sequence_inputs = []
    targets = []

    for i in range(size):
        frequency_index = i % 4
        sigma = 0.05 + 0.15 * ((i % 10) / 10)
        example = create_example(frequency_index, sigma)

        mlp_inputs.append(build_mlp_input(example))
        sequence_inputs.append(build_sequence_input(example))
        targets.append(torch.tensor(example["clean_samples"], dtype=torch.float32))

    return (
        torch.stack(mlp_inputs),
        torch.stack(sequence_inputs),
        torch.stack(targets),
    )


def train_model(model, inputs, targets, epochs=30):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for _ in range(epochs):
        prediction = model(inputs)
        loss = loss_fn(prediction, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss.item()


def run_training():
    mlp_x, seq_x, y = build_dataset()

    results = {
        "MLP": train_model(MLPModel(), mlp_x, y),
        "RNN": train_model(RNNModel(), seq_x, y),
        "LSTM": train_model(LSTMModel(), seq_x, y),
    }

    return results


if __name__ == "__main__":
    final_results = run_training()

    print("Final MSE Loss Results:")
    for model_name, loss in final_results.items():
        print(f"{model_name}: {loss:.6f}")