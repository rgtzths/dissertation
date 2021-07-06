import json
from matplotlib import pyplot as plt

def plot_model_history(model_history, destination_folder):
    
    n_epochs = len(model_history['loss'])

    fig = plt.figure()
    plt.plot(range(1, n_epochs+1), model_history['loss'], label="Training Loss")
    plt.plot(range(1, n_epochs+1), model_history['val_loss'], label="Validation Loss")

    plt.xlabel("Nº Epochs")
    plt.ylabel("Categorical Cross Entropy")
    plt.title("Loss of the model")
    plt.legend()
    print(fig.dpi)
    fig.savefig(destination_folder+"loss.png", dpi=300.0, bbox_inches='tight', orientation="landscape")

    fig = plt.figure()
    plt.plot(range(1, n_epochs+1), model_history['accuracy'], label="Training Accuracy")
    plt.plot(range(1, n_epochs+1), model_history['val_accuracy'], label="Validation Accuracy")

    plt.xlabel("Nº Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy of the model")
    plt.legend()

    fig.savefig(destination_folder+"acc.png", dpi=300.0, bbox_inches='tight', orientation="landscape")

    fig = plt.figure()
    plt.plot(range(1, n_epochs+1), model_history['matthews_correlation'], label="Training MCC")
    plt.plot(range(1, n_epochs+1), model_history['val_matthews_correlation'], label="Validation MCC")

    plt.xlabel("Nº Epochs")
    plt.ylabel("Matthews Correlation Coeficient")
    plt.title("Matthews Correlation Coeficient of the model")
    plt.legend()

    fig.savefig(destination_folder+"mcc.png", dpi=300.0, bbox_inches='tight', orientation="landscape")


