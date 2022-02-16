
from matplotlib import pyplot as plt
import json


font = {'family' : 'monospace',
        'size'   : 14,
        #'weight': 'bold'
        }

plt.rc('font', **font)

plt.rcParams['figure.figsize'] = [6, 6]

def plot_model_history_classification(model_history, appliance, model):
    
    n_epochs = len(model_history['loss'])

    fig = plt.figure()
    plt.plot(range(1, n_epochs+1), model_history['loss'], label="Training Loss")
    plt.plot(range(1, n_epochs+1), model_history['val_loss'], label="Validation Loss")

    plt.xlabel("Nº Epochs", weight="bold")
    plt.ylabel("Categorical Cross Entropy", weight="bold")
    #plt.title( model + "'s loss for appliance " + appliance.replace("_", " "))
    plt.legend()
    plt.ylim([0, 1])
    plt.subplots_adjust(left=0.13, right=0.99, top=0.97, bottom=0.09)
    plt.show()


model = "DeepGRU"
appliances = ["fridge", "washing_machine", "kettle", "microwave", "dish_washer"]
path = "regression-model1"

for appliance in appliances:
    model1 = json.load(open("/home/user/final_results/"+ path+"/history/"+model+"/history_"+appliance+".json"))

    plot_model_history_classification(model1, appliance, model)