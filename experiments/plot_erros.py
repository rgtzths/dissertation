import json
from matplotlib import pyplot as plt

filename = "/home/user/Desktop/ZiTh0s/Uni/Tese/results/trial_6/history/history_microwave_ResNet.json"

history = json.load(open(filename))

n_epochs = len(history['loss'])

plt.plot(range(1, n_epochs+1), history['loss'], label="Training Loss")
plt.plot(range(1, n_epochs+1), history['val_loss'], label="Validation Loss")

plt.xlabel("Nº Epochs")
plt.ylabel("Categorical Cross Entropy")
plt.title("Loss of the model")
plt.legend()

plt.show()

plt.plot(range(1, n_epochs+1), history['accuracy'], label="Training Accuracy")
plt.plot(range(1, n_epochs+1), history['val_accuracy'], label="Validation Accuracy")

plt.xlabel("Nº Epochs")
plt.ylabel("Accuracy")
plt.title("Accuracy of the model")
plt.legend()

plt.show()

plt.plot(range(1, n_epochs+1), history['matthews_correlation'], label="Training MCC")
plt.plot(range(1, n_epochs+1), history['val_matthews_correlation'], label="Validation MCC")

plt.xlabel("Nº Epochs")
plt.ylabel("Matthews Correlation Coeficient")
plt.title("Matthews Correlation Coeficient of the model")
plt.legend()

plt.show()