import numpy as np
import torch
from create_sequences import create_sequences


def prepare_data(seq_length):
    np.random.seed(0)
    torch.manual_seed(0)

    # Generate synthetic sine wave data
    t = np.linspace(0, 100, 1000)
    data = np.sin(t)

    # Create sequences
    X, y = create_sequences(data, seq_length)

    # Convert to PyTorch tensors
    trainX = torch.tensor(X[:, :, None], dtype=torch.float32)
    trainY = torch.tensor(y[:, None], dtype=torch.float32)

    return data, trainX, trainY
