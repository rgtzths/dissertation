import json
from matplotlib import pyplot as plt

filename = "/home/user/thesis_results/history/history_heatpump_GRU_DWT_500_1.json"

history = json.load(open(filename))

n_epochs = len(history['loss'])

plt.plot(range(1, n_epochs+1), history['loss'], label="Training Loss")
plt.plot(range(1, n_epochs+1), history['val_loss'], label="Validation Loss")

plt.xlabel("Nº Epochs")
plt.ylabel("Binary Cross Entropy")
plt.title("Loss of the model")
plt.legend()

plt.show()

plt.plot(range(1, n_epochs+1), history['accuracy'], label="Training MCC")
plt.plot(range(1, n_epochs+1), history['val_accuracy'], label="Validation MCC")

plt.xlabel("Nº Epochs")
plt.ylabel("Matthews Correlation Coeficient")
plt.title("Matthews Correlation Coeficient of the model")
plt.legend()

plt.show()