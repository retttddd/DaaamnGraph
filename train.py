import torch
import torch.nn as nn
from model import LSTMModel
from dataset import prepare_data


def train_model(seq_length, num_epochs=1000, lr=0.01):
    data, trainX, trainY = prepare_data(seq_length)

    # Initialize model, loss, and optimizer
    model = LSTMModel(input_dim=1, hidden_dim=100, layer_dim=1, output_dim=1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        outputs = model(trainX)
        optimizer.zero_grad()
        loss = criterion(outputs, trainY)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

    return model, data, trainX
