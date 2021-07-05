from nilmtk import DataSet
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=RuntimeWarning)

'''
    Plot of Different Readings from the same house
'''
timeperiod=("2021-6-1","2021-6-7")
house=1
app="fridge"

dataset = DataSet('../../datasets/withus_h5/withus.h5')

dataset.set_window(start=timeperiod[0],end=timeperiod[1])

mains = next(dataset.buildings[1].elec.mains().load())
app_df = next(dataset.buildings[1].elec[app].load())
#plt.plot(mains.index, mains["power"]["apparent"], label="Mains Energy")
plt.plot(app_df.index, app_df["power"]["active"], label=app + " Energy")

mains = next(dataset.buildings[2].elec.mains().load())
app_df = next(dataset.buildings[2].elec[app].load())
#plt.plot(mains.index, mains["power"]["apparent"], label="Mains Energy")
plt.plot(app_df.index, app_df["power"]["active"], label=app + " Energy")

mains = next(dataset.buildings[4].elec.mains().load())
app_df = next(dataset.buildings[4].elec[app].load())
#plt.plot(mains.index, mains["power"]["apparent"], label="Mains Energy")
plt.plot(app_df.index, app_df["power"]["active"], label=app + " Energy")


plt.xlabel("Time")
plt.ylabel("Power")
plt.title("Mains and " + app + " energy consumption from house" + str(house))
plt.legend()

plt.show()