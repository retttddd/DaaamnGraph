from train import train_model
from results import plot_results

if __name__ == "__main__":
    seq_length = 200
    model, data, trainX = train_model(seq_length)
    plot_results(model, data, trainX, seq_length)
    #1Iteration