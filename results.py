import numpy as np
import matplotlib.pyplot as plt

def plot_results(model, data, trainX, seq_length):
    model.eval()
    predicted = model(trainX).detach().numpy()

    # Adjust original data and prediction for plotting
    original = data[seq_length:]  # Original data from the end of the first sequence
    time_steps = np.arange(seq_length, len(data))  # Corresponding time steps

    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, original, label='Original Data')
    plt.plot(time_steps, predicted, label='Predicted Data', linestyle='--')
    plt.title('LSTM Model Predictions vs. Original Data')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
