import json
from matplotlib import pyplot as plt

filename_1 = "/home/user/Desktop/ZiTh0s/Uni/Tese/results/trial_7/history/history_carcharger_ResNet.json"
filename_2 = "/home/user/Desktop/ZiTh0s/Uni/Tese/results/trial_8/history/history_carcharger_ResNet.json"

history_1 = json.load(open(filename_1))
history_2 = json.load(open(filename_2))

n_epochs = len(history_1['loss'])

plt.plot(range(1, n_epochs+1), history_1['matthews_correlation'], label="Training MCC Before")
plt.plot(range(1, n_epochs+1), history_1['val_matthews_correlation'], label="Validation MCC Before")

plt.plot(range(1, n_epochs+1), history_2['matthews_correlation'], label="Training MCC After")
plt.plot(range(1, n_epochs+1), history_2['val_matthews_correlation'], label="Validation MCC After")

plt.xlabel("Nº Epochs")
plt.ylabel("Matthews Correlation Coeficient")
plt.title("Matthews Correlation Coeficient of the model")
plt.legend()

plt.show()