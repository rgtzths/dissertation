from matplotlib import pyplot as plt
import json

font = {'family' : 'monospace',
        'size'   : 14,
        #'weight': 'bold'
        }

plt.rc('font', **font)

plt.rcParams['figure.figsize'] = [6, 6]

def compare_errors_classification(model_history_1, model_history_2, appliance, model):

    n_epochs = len(model_history_1['loss'])

    plt.plot(range(1, n_epochs+1), model_history_1['loss'], label="Training Loss-Previous Model")
    plt.plot(range(1, n_epochs+1), model_history_1['val_loss'], label="Validation Loss-Previous Model")

    plt.plot(range(1, n_epochs+1), model_history_2['loss'], label="Training Loss-New Model")
    plt.plot(range(1, n_epochs+1), model_history_2['val_loss'], label="Validation Loss-New Model")

    plt.xlabel("Nº Epochs", weight="bold")
    plt.ylabel("Categorical Cross Entropy", weight="bold")
    #plt.title("Comparison of the " + model + " architecture's loss for " + appliance, weight="bold", size=12)
    plt.ylim([0, 2])
    plt.subplots_adjust(left=0.14, right=0.99, top=0.97, bottom=0.09)
    plt.legend()

    plt.show()

def plot_bars(name, f1_values):
    legends = ("ResNet", "DeepGRU", "MLP", "MLP_Raw", "AE", "LSTM", "Rectangles")
    titles = ["Fridge", "Microwave", "Kettle", "Washing Machine", "Dishwasher"]
    width = 0.25
    bars_idx = [width*x[0] for x in enumerate(legends)]
    colors = ["tab:cyan", "tab:olive", "tab:brown", "tab:blue", "tab:orange", "tab:red", "tab:green"]

    fig = plt.figure(figsize=(15, 10))
    ax1 = plt.subplot(2,3,1)
    ax2 = plt.subplot(2,3,2)
    ax3 = plt.subplot(2,3,3)
    ax4 = plt.subplot(2,2,3)
    ax5 =  plt.subplot(2,2,4)
    axs = [ax1, ax2, ax3, ax4, ax5]

    ###Plot F1-Score

    for idx, title in enumerate(titles[0:3]):
        bars = []
        for model_idx in range(0, len(legends)):
            bars.append(axs[idx].bar(bars_idx[model_idx], round(f1_values[idx][model_idx], 2), width, color = colors[model_idx]))

        axs[idx].set_title(title)
        axs[idx].set_ylim([0, 1])
        for bar in bars:
            for p in bar:
                height = p.get_height()
                axs[idx].annotate('{}'.format(height),
                    xy=(p.get_x() + p.get_width() / 2, 0.5),
                    xytext=(0, -15), # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', rotation=90)
        
        axs[idx].tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False)

    for idx, title in enumerate(titles[3:]):
        bars = []
        for model_idx in range(0, len(legends)):
            bars.append(axs[idx+3].bar(bars_idx[model_idx], round(f1_values[idx+3][model_idx], 2), width, color = colors[model_idx]))

        axs[idx+3].set_title(title)
        axs[idx+3].set_ylim([0, 1])
        for bar in bars:
            for p in bar:
                height = p.get_height()
                axs[idx+3].annotate('{}'.format(height),
                    xy=(p.get_x() + p.get_width() / 2, 0.5),
                    xytext=(0, -15), # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', rotation=90)
        
        axs[idx+3].tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False)

    fig.legend(bars, legends, prop={'size': 15})

    for idx, ax in enumerate(axs):
        ax.set(xlabel='', ylabel="")

    axs[0].set(xlabel='', ylabel="F1-Score")
    axs[3].set(xlabel='', ylabel="F1-Score")

    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)

    plt.subplots_adjust(left=0.05, right=0.95, top=0.78, bottom=0.05)

    plt.show()


model = "DeepGRU"
appliances = ["fridge", "washing_machine", "kettle", "microwave", "dish_washer"]
paths = ("regression-model1", "regression-model2")

#for appliance in appliances:
    #model1 = json.load(open("/home/user/final_results/"+ paths[0]+"/history/"+model+"/history_"+appliance+".json"))
    #model2 = json.load(open("/home/user/final_results/"+ paths[1]+"/history/"+ model +"/history_"+appliance+".json"))

    #compare_errors_classification(model1, model2, appliance.replace("_", " "), model)


f1_values = [
    [0.5725135216, 0.6896744186, 0.661183831469, 0.691727362273, 0.82, 0.74, 0.74],
    [0.8391866913, 0.8128135855, 0.686440677966, 0.821013133208, 0.26, 0.13, 0.21],
    [0.9564526803, 0.95516052, 0.937165775401, 0.958223162348, 0.93, 0.93, 0.7],
    [0.6765280717, 0.7136939257, 0.469580089748, 0.747661365762, 0.27, 0.03, 0.27],
    [0.7308836862, 0.7340762951, 0.668752333501, 0.743435818875,0.44 , 0.08, 0.74]
]

plot_bars("F1-Score comparison using the UKDALE dataset.", f1_values)