import json
from matplotlib import pyplot as plt

def plot_model_history_classification(model_history, destination_folder):
    
    n_epochs = len(model_history['loss'])

    fig = plt.figure()
    plt.plot(range(1, n_epochs+1), model_history['loss'], label="Training Loss")
    plt.plot(range(1, n_epochs+1), model_history['val_loss'], label="Validation Loss")

    plt.xlabel("Nº Epochs")
    plt.ylabel("Categorical Cross Entropy")
    plt.title("Evolution of the loss of the model")
    plt.legend()
    fig.savefig(destination_folder+"loss.png", dpi=300.0, bbox_inches='tight', orientation="landscape")

    fig = plt.figure()
    plt.plot(range(1, n_epochs+1), model_history['accuracy'], label="Training Accuracy")
    plt.plot(range(1, n_epochs+1), model_history['val_accuracy'], label="Validation Accuracy")

    plt.xlabel("Nº Epochs")
    plt.ylabel("Accuracy")
    plt.title("Evolution of the accuracy of the model")
    plt.legend()

    fig.savefig(destination_folder+"acc.png", dpi=300.0, bbox_inches='tight', orientation="landscape")

    fig = plt.figure()
    plt.plot(range(1, n_epochs+1), model_history['matthews_correlation'], label="Training MCC")
    plt.plot(range(1, n_epochs+1), model_history['val_matthews_correlation'], label="Validation MCC")

    plt.xlabel("Nº Epochs")
    plt.ylabel("Matthews Correlation Coeficient")
    plt.title("Evolution of the MCC of the model")
    plt.legend()

    fig.savefig(destination_folder+"mcc.png", dpi=300.0, bbox_inches='tight', orientation="landscape")

def plot_model_history_regression(model_history, destination_folder):
    
    n_epochs = len(model_history['loss'])

    fig = plt.figure()
    plt.plot(range(1, n_epochs+1), model_history['loss'], label="Training Loss")
    plt.plot(range(1, n_epochs+1), model_history['val_loss'], label="Validation Loss")

    plt.xlabel("Nº Epochs")
    plt.ylabel("Mean Squared Error")
    plt.title("Evolution of the loss of the model")
    plt.legend()
    fig.savefig(destination_folder+"loss.png", dpi=300.0, bbox_inches='tight', orientation="landscape")

    fig = plt.figure()
    plt.plot(range(1, n_epochs+1), model_history['root_mean_squared_error'], label="Training RMSE")
    plt.plot(range(1, n_epochs+1), model_history['val_root_mean_squared_error'], label="Validation RMSE")

    plt.xlabel("Nº Epochs")
    plt.ylabel("Root Mean Squared Error")
    plt.title("Evolution of the RMSE of the model")
    plt.legend()

    fig.savefig(destination_folder+"rmse.png", dpi=300.0, bbox_inches='tight', orientation="landscape")

    fig = plt.figure()
    plt.plot(range(1, n_epochs+1), model_history['mean_absolute_error'], label="Training MAE")
    plt.plot(range(1, n_epochs+1), model_history['val_mean_absolute_error'], label="Validation MAE")

    plt.xlabel("Nº Epochs")
    plt.ylabel("Mean Absolute Error")
    plt.title("Evolution of the MAE of the model")
    plt.legend()

    fig.savefig(destination_folder+"mae.png", dpi=300.0, bbox_inches='tight', orientation="landscape")


