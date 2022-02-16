import numpy as np
import matplotlib.pyplot as plt

font = {'family' : 'DejaVu Sans',
        'size'   : 15}

plt.rc('font', **font)

gaps = [1, 3, 5, 10, 20]

average = [0.267602215, 1.230727905, 0.9491128063, 1.774268645, 4.209748932]
linear = [0.2676150777, 1.292466853, 1.022338087, 1.422150876, 3.224181792]
previous = [0.3609654279, 1.654614281, 1.664577828, 2.40339172, 5.03958606]



plt.plot(gaps,average,  marker='o', linestyle='--', label="Our Algorithm", linewidth=2)
plt.plot(gaps, previous, marker='x', linestyle='--', label="Traces-Previous", linewidth=2)
plt.plot(gaps, linear, marker='v', linestyle='--', label="Traces-Linear", linewidth=2)


plt.xlabel("Gap Size")
plt.ylabel("RMSE")
plt.xticks(ticks=gaps, labels=gaps)
#plt.title("Average error obtained by the algorithms when filling gaps")
plt.legend()

plt.show()