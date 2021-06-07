
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=RuntimeWarning)



'''
    Energia aggregada e dos dispositivos.
'''

beginning = pd.to_datetime('2020-12-02T11:00')
end = pd.to_datetime('2020-12-02T14:00')

app_file = "../../datasets/avEiro/house_1/mains/power.csv"

df = pd.read_csv(app_file)

df.index = pd.to_datetime(df["time"])

beginning_index = df.index.get_loc(beginning, method="nearest")

end_index = df.index.get_loc(end, method="nearest")


app_file = "../../datasets/avEiro/house_1/heatpump/power.csv"

df2 = pd.read_csv(app_file)

df2.index = pd.to_datetime(df2["time"])

beginning_index_2 = df2.index.get_loc(beginning, method="nearest")

end_index_2 = df2.index.get_loc(end, method="nearest")


app_file = "../../datasets/avEiro/house_1/carcharger/power.csv"

df3 = pd.read_csv(app_file)

df3.index = pd.to_datetime(df3["time"])

beginning_index_3 = df3.index.get_loc(beginning, method="nearest")

end_index_3 = df3.index.get_loc(end, method="nearest")



plt.plot(df.index[beginning_index:end_index], df["value"][beginning_index:end_index], label="Apparent Energy")
plt.plot(df2.index[beginning_index_2:end_index_2], df2["value"][beginning_index_2:end_index_2], label="Heat pump Energy")
plt.plot(df3.index[beginning_index_3:end_index_3], df3["value"][beginning_index_3:end_index_3], label="Car charger Energy")

plt.xlabel("Time")
plt.ylabel("Power")
plt.title("Household energy consumption")
plt.legend()

plt.show()
#
#
#'''
#    Energia aggregada e voltagem
#'''
#
#beginning = pd.to_datetime('2020-12-02T03:30')
#end = pd.to_datetime('2020-12-02T04:00')
#
#app_file = "../../datasets/avEiro/house_1/mains/vrms.csv"
#
#df4 = pd.read_csv(app_file)
#
#df4.index = pd.to_datetime(df4["time"])
#
#beginning_index = df.index.get_loc(beginning, method="nearest")
#
#end_index = df.index.get_loc(end, method="nearest")
#
#beginning_index_4 = df4.index.get_loc(beginning, method="nearest")
#
#end_index_4 = df4.index.get_loc(end, method="nearest")
#
#
#plt.plot(df.index[beginning_index:end_index], df["value"][beginning_index:end_index], label="Apparent Energy")
#plt.plot(df4.index[beginning_index_4:end_index_4], df4["value"][beginning_index_4:end_index_4], label="Voltage")
#
#plt.xlabel("Time")
#plt.ylabel("Power")
#plt.title("Aggregated Power and Voltage Readings")
#plt.legend()

#plt.show()


#'''
#    Diferença de utilização ao longo do tempo.
#'''
#
#beginning = pd.to_datetime('2020-10-12T12:30')
#end = pd.to_datetime('2020-10-19T12:30')
#
#beginning_index_1 = df2.index.get_loc(beginning, method="nearest")
#
#end_index_1 = df2.index.get_loc(end, method="nearest")
#
#beginning = pd.to_datetime('2020-12-12T12:30')
#end = pd.to_datetime('2020-12-19T12:30')
#
#beginning_index_2 = df2.index.get_loc(beginning, method="nearest")
#
#end_index_2 = df2.index.get_loc(end, method="nearest")
#
#plt.plot(df2.index[beginning_index_1:end_index_1], df2["value"][beginning_index_1:end_index_1], label="October Readings")
#
#plt.xlabel("Time")
#plt.ylabel("Power")
#plt.title("October Readings from 1 week - Heat pump")
#plt.legend()
#
#plt.show()
#
#plt.plot(df2.index[beginning_index_2:end_index_2], df2["value"][beginning_index_2:end_index_2], label="December readings")
#
#plt.xlabel("Time")
#plt.ylabel("Power")
#plt.title("December Readings from 1 week - Heat pump")
#plt.legend()
#
#plt.show()