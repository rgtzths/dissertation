import pandas as pd
import matplotlib.pyplot as plt


font = {'family' : 'monospace',
        'size'   : 14,
        #'weight': 'bold'
        }

plt.rc('font', **font)

plt.rcParams['figure.figsize'] = [6, 6]

input_file = "./original.csv"
timestep = 6

begining = pd.to_datetime('2014-01-01')
end = pd.to_datetime('2014-01-02')

original = pd.read_csv(input_file, sep=",", header=None)
original.columns = ["time", "power"]
original.index = pd.to_datetime(original["time"], format='%Y-%m-%d %H:%M:%S')        
original = original.drop("time", 1)
begining_index = original.index.get_loc(begining, method="nearest")
end_index = original.index.get_loc(end, method="nearest")
original = original[begining_index:end_index]


plt.plot(original.index, original["power"])

plt.ylabel("Active power")
# disabling xticks by Setting xticks to an empty list
plt.xticks([])  
  
# disabling yticks by setting yticks to an empty list
plt.yticks([])
plt.subplots_adjust(left=0.13, right=0.99, top=0.97, bottom=0.09)
#plt.title("Fridges signature")

plt.show()