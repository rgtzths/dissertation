import numpy as np
import matplotlib.pyplot as plt

font = {'family' : 'DejaVu Sans',
        'size'   : 15}

plt.rc('font', **font)

def plot_bars(name, mae_values, rmse_values):
    legends = ("DAE", "Seq2Seq", "Seq2Point", "ResNet", "DeepGRU", "MLP_RAW", "MLP")
    titles = ["Fridge", "Microwave", "Kettle", "Washing Machine", "Dishwasher"]
    width = 0.25
    bars_idx = [width*x[0] for x in enumerate(legends)]
    colors = ["tab:cyan", "tab:olive", "tab:brown", "tab:blue", "tab:orange", "tab:red", "tab:green"]

    fig, axs = plt.subplots(2, 5)
    fig.suptitle(name)
    fig.set_figheight(10)
    fig.set_figwidth(21)

    ###Plot MAE

    for idx, title in enumerate(titles):
        bars = []
        for model_idx in range(0, len(legends)):
            bars.append(axs[0, idx].bar(bars_idx[model_idx], round(mae_values[idx][model_idx], 2), width, color = colors[model_idx]))

        axs[0, idx].set_title(title)

        for bar in bars:
            for p in bar:
                height = p.get_height()
                axs[0,idx].annotate('{}'.format(height),
                    xy=(p.get_x() + p.get_width() / 2, height/2),
                    xytext=(0, -15), # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', rotation=90)
        
        axs[0,idx].tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False)

    ###Plot RMSE

    for idx, title in enumerate(titles):
        bars = []
        for model_idx in range(0, len(legends)):
            bars.append(axs[1, idx].bar(bars_idx[model_idx], round(rmse_values[idx][model_idx], 2), width, color = colors[model_idx]))

        axs[1, idx].set_title(title)

        for bar in bars:
            for p in bar:
                height = p.get_height()
                axs[1,idx].annotate('{}'.format(height),
                    xy=(p.get_x() + p.get_width() / 2, height/2),
                    xytext=(0, -15), # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', rotation=90)
        
        axs[1,idx].tick_params(
                axis='x',          # changes apply to the x-axis
                which='both',      # both major and minor ticks are affected
                bottom=False,      # ticks along the bottom edge are off
                top=False,         # ticks along the top edge are off
                labelbottom=False)


    fig.legend(bars, legends, prop={'size': 15})

    for idx, ax in enumerate(axs.flat):
        ax.set(xlabel='', ylabel="")

    axs[0,0].set(xlabel='', ylabel="Average Mean Absolute Error")
    axs[1,0].set(xlabel='', ylabel="Average Root Mean Squared Error")

    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)



    plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.05)
    plt.savefig(name.replace(" ", "_")+'.png', bbox_inches='tight', dpi=300)

    plt.show()

title = 'Average MAE & RMSE - Same Dataset'
mae_values = [
    [51.51092377, 61.69079056, 16.42436409, 43.34014511, 44.31448898, 31.74103394,  47.77838478],
    [717.0412476, 707.214624, 265.5007141, 434.0737915, 371.8857117, 373.4557434, 1500.522852],
    [1251.037939, 1253.87312, 254.8040924, 551.7960815, 367.0702576, 305.5707275, 1121.649548],
    [392.4761841, 380.3637024, 244.0782928, 318.288092, 214.2493103, 243.5935181, 327.9631165],
    [727.1795288, 753.4077271, 218.558136, 286.4516479, 279.7287567, 307.7263306, 377.7758057]
]

rmse_values = [
    [69.27700224, 74.29395575, 32.97287176, 55.90519376, 55.26212502, 47.26126088, 69.06433711],
    [766.296881, 928.3852705, 383.3901904, 581.2625942, 488.0670474, 476.9341224, 1539.873133],
    [1819.589234, 1774.358807, 563.5641875, 824.8958934, 716.2083547, 672.8871842, 1763.477681],
    [704.6994047, 696.7616039, 504.4040574, 484.5526016, 422.0832226, 478.3074405, 592.8049443],
    [1165.572236, 1166.128681, 444.2883396, 506.9413443, 550.7723442, 525.9424766, 656.9920213]
]

plot_bars(title, mae_values, rmse_values)

title = 'Average MAE & RMSE - Transfer Learning Between Datasets'
mae_values = [
    [47.829772, 54.02256966, 15.50751424, 40.35756079, 37.42340962, 28.93016211, 48.68989786],
    [626.0854422, 600.8372263, 288.7084879, 608.5081552, 331.6805021, 326.7989326, 3044.248003],
    [1207.021479, 1100.210118, 350.2263738, 947.2942505, 449.1773765, 390.8940568, 1089.537817],
    [463.7116253, 450.6375545, 322.0512038, 360.446106, 311.6060861, 318.4983955, 347.8786926],
    [609.4427338, 735.4994976, 263.7902738, 362.0368195, 341.3121326, 320.1893381, 369.3349504]
]

rmse_values = [
    [65.11588099, 67.28948536, 29.93280216, 50.72503441, 50.82593774, 43.51749147, 67.18377135],
    [841.5908189, 782.1599922, 417.0877794, 694.5148334, 467.0898824, 478.7743542, 3211.502516],
    [1541.648347, 1520.584373, 563.1689301, 1135.183903, 723.9838063, 678.3254626, 1439.636688],
    [759.3181511, 747.9994306, 549.9692392, 633.6150365, 512.8733676, 526.2903361, 584.22491],
    [969.5344386, 1120.09767, 466.8872153, 606.0599035, 544.4495828, 561.0596311, 652.0173601]
]

plot_bars(title, mae_values, rmse_values)