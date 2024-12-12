import numpy as np
import torch
from create_sequences import create_sequences
import pandas as pd

def extract_data(filename='data/GOOGL.csv'):
    df = pd.read_csv(filename)
    return df

def prepare_data(seq_length):
    np.random.seed(0)
    torch.manual_seed(0)

    data = extract_data()

    close = data['Close'].values[:500]

    # Create sequences
    X, y = create_sequences(close, seq_length)

    # Convert to PyTorch tensors
    trainX = torch.tensor(X[:, :, None], dtype=torch.float32)
    trainY = torch.tensor(y[:, None], dtype=torch.float32)

    return close, trainX, trainY
