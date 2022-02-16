import numpy as np
import matplotlib.pyplot as plt

font = {'family' : 'DejaVu Sans',
        'size'   : 12}

plt.rc('font', **font)

def plot_bars(name, mc_values, legends):
    width = 0.1
    bars_idx = [0.15*x[0] for x in enumerate(legends)]
    colors = ["tab:cyan", "tab:olive", "tab:brown", "tab:orange", "tab:red", "tab:green", "tab:blue",]

    fig, axs = plt.subplots(1, 1)
    fig.set_figheight(6)
    fig.set_figwidth(15)
    axs.set_axisbelow(True)
    axs.grid(color='gray', linestyle='dashed', linewidth=0.5)

    bars = []
    for model_idx in range(0, len(legends)):
        bars.append(axs.bar(bars_idx[model_idx], round(mc_values[model_idx], 2), width, color = colors[model_idx]))

    for bar in bars:
        for p in bar:
            height = p.get_height()
            axs.annotate('{}'.format(height),
                xy=(p.get_x() + p.get_width() / 2, 
                    height*2 if height < 0.1 else height/2 + 0.01),
                xytext=(0, -15), # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom', rotation=0)
        
    axs.set_xticks(bars_idx)
    axs.set_xticklabels(legends) 
    plt.subplots_adjust(left=0.03, right=0.98, top=0.97, bottom=0.05)
    plt.savefig(name.replace(" ", "_")+'.png', bbox_inches='tight', dpi=300)

    plt.show()

#From scratch
legends = ("DPWC", "LSI", "Word2Vec", "GloVe", "fastText", "RoBERTa - Corpus", "RoBERTa - Task")

mc_values = [0.66, 0.42, 0.64, 0.56, 0.59, 0.1, 0.94]

iot_values = [0.55, 0.53, 0.35, 0.31, 0.63, 0.58, 0.03]

plot_bars("from_scratch_mc", mc_values, legends)

plot_bars("from_scratch_iot", iot_values, legends)

#pre-trained
legends = ("DPWC", "Word2Vec", "GloVe", "fastText", "fastText - online", "RoBERTa - Corpus", "RoBERTa - Task(online)")

mc_values = [0.66, 0.78, 0.74, 0.81, 0.60, 0.14, 0.95]

iot_values = [0.55, 0.53, 0.35, 0.31, 0.63, 0.32, 0.12]

plot_bars("pretrained_mc", mc_values, legends)

plot_bars("pretrained_iot", iot_values, legends)