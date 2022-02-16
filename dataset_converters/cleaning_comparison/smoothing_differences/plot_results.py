import numpy as np
import matplotlib.pyplot as plt

font = {'family' : 'DejaVu Sans',
        'size'   : 15,
        'weight': 'bold'
        }

plt.rc('font', **font)
plt.rc('axes', labelsize=20)

#plt.rcParams['figure.figsize'] = [12, 10]


std = [1, 1.5, 3, 9, 27]

ema3 = [0.2457106473, 0.3700213315, 0.738537122, 2.214694104, 6.714831061]
ema5 = [0.1901247556, 0.2866389258, 0.5713613539, 1.713030335, 5.202244392]
ema9 = [0.14151801, 0.2136781529, 0.4247152027, 1.273991933, 3.879791407]
sma3 = [0.2455978804, 0.3698747582, 0.7386191602, 2.215741787, 6.717505285]
sma5 = [0.1899816142, 0.2865596341, 0.5715503422, 1.713798827, 5.199662191]
sma9 = [0.141488777, 0.213613029, 0.4246339009, 1.271941919, 3.873436836]
median3 = [0.1987443868, 0.2693537549, 0.4732564031, 1.319438861, 3.715104983]
median5 = [0.1621124608, 0.2069839908, 0.3141940554, 0.7356664801, 1.877941065]
median9 = [0.149491259, 0.1891299396, 0.2777186539, 0.6142714779, 1.486725403]


plt.figure(figsize=(13, 10))
plt.plot(std, ema3,  marker='o', linestyle='--', label="Window: 3",  linewidth=2)
plt.plot(std, ema5, marker='x', linestyle='--', label="Window: 5",  linewidth=2)
plt.plot(std, ema9, marker='v', linestyle='--', label="Window: 9", linewidth=2)

plt.xlabel("Standard deviation of the Gaussian")
plt.ylabel("RMSE")
plt.xticks(ticks=std, labels=std)
plt.ylim([0, 7])
#plt.title("Average error obtained by the EWMA when correcting noise.")
plt.subplots_adjust(left=0.06, right=0.99, top=0.97, bottom=0.09)
#plt.legend(loc=2, prop={'size': 14})
plt.legend()

plt.show()

plt.figure(figsize=(13, 10))
plt.plot(std, sma3,  marker='o', linestyle='--', label="Window: 3",  linewidth=2)
plt.plot(std, sma5, marker='x', linestyle='--', label="Window: 5",  linewidth=2)
plt.plot(std, sma9, marker='v', linestyle='--', label="Window: 9",  linewidth=2)

plt.xlabel("Standard deviation of the Gaussian")
plt.ylabel("RMSE")
plt.xticks(ticks=std, labels=std)
plt.ylim([0, 7])
#plt.title("Average error obtained by the SMA when correcting noise.")
plt.subplots_adjust(left=0.06, right=0.99, top=0.97, bottom=0.09)
plt.legend()

plt.show()

plt.figure(figsize=(13, 10))
plt.plot(std, median3,  marker='o', linestyle='--', label="Window: 3",  linewidth=2)
plt.plot(std, median5, marker='x', linestyle='--', label="Window: 5",  linewidth=2)
plt.plot(std, median9, marker='v', linestyle='--', label="Window: 9",  linewidth=2)

plt.xlabel("Standard deviation of the Gaussian")
plt.ylabel("RMSE")
plt.xticks(ticks=std, labels=std)
plt.ylim([0, 7])
#plt.title("Average error obtained by the Median Filter when correcting noise.")
plt.subplots_adjust(left=0.06, right=0.99, top=0.97, bottom=0.09)
plt.legend()
plt.show()

plt.figure(figsize=(16, 10))
plt.plot(std, sma5,  marker='o', linestyle='--', label="SMA",  linewidth=2)
plt.plot(std, median3, marker='x', linestyle='--', label="Median Filter",  linewidth=2)
plt.plot(std, ema9, marker='v', linestyle='--', label="EWMA",  linewidth=2)

plt.xlabel("Standard deviation of the Gaussian")
plt.ylabel("RMSE")
plt.xticks(ticks=std, labels=std)
plt.ylim([0, 7])
plt.subplots_adjust(left=0.06, right=0.99, top=0.97, bottom=0.09)
#plt.title("Average error obtained by the different smoothing techiques when correcting noise.")
plt.legend()

plt.show()