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
#ampds = DataSet('../../datasets/ampds2/AMPds2.h5')
#
#df = next(ampds.buildings[1].elec.mains().load(physical_quantity="power"))
#
#beginning_index = df.index.get_loc(beginning, method="nearest")
#
#end_index = df.index.get_loc(end, method="nearest")
#
#plt.plot(df.index[beginning_index:end_index], df["power"]["active"][beginning_index:end_index], label="Active Power")
#plt.plot(df.index[beginning_index:end_index],df["power"]["apparent"][beginning_index:end_index], label="Apparent Power")
#plt.plot(df.index[beginning_index:end_index], df["power"]["reactive"][beginning_index:end_index], label="Reactive Power")
##plt.plot(df.index[beginning_index:end_index], df["power factor"]["apparent"][beginning_index:end_index], label="Apparent Power")
#
#plt.xlabel("Time")
#plt.ylabel("Power")
#plt.title("Multiple Aggregated Readings")
#plt.legend()
#
#plt.show()


'''
    Plot of the same appliance from different houses
'''
ukdale = DataSet('../../datasets/ukdale/ukdale.h5')
beginning = pd.to_datetime('2013-05-20T00:00')
end = pd.to_datetime('2013-10-11T00:00')

b1 = next(ukdale.buildings[2].elec["dish washer"].load())

beginning_index_1 = b1.index.get_loc(beginning, method="nearest")

end_index_1 = b1.index.get_loc(end, method="nearest")


#beginning = pd.to_datetime('2013-06-02T12:30')
#end = pd.to_datetime('2013-06-02T20:00')
#
#b2 = next(ukdale.buildings[2].elec["fridge"].load())
#
#beginning_index_2 = b2.index.get_loc(beginning, method="nearest")
#
#end_index_2 = b2.index.get_loc(end, method="nearest")
#
#
#beginning = pd.to_datetime('2014-07-02T12:30')
#end = pd.to_datetime('2014-07-02T20:00')
#
#b5 = next(ukdale.buildings[5].elec["fridge"].load())
#
#beginning_index_5 = b5.index.get_loc(beginning, method="nearest")
#
#end_index_5 = b5.index.get_loc(end, method="nearest")


plt.plot(b1.index[beginning_index_1:end_index_1], b1["power"]["active"][beginning_index_1:end_index_1], label="Building 1")
#plt.plot(b2.index[beginning_index_2:end_index_2],b2["power"]["active"][beginning_index_2:end_index_2], label="Building 2")
#plt.plot(b2.index[beginning_index_2:end_index_2], b5["power"]["active"][beginning_index_5:end_index_5][0: b2.index[beginning_index_2:end_index_2].shape[0] ], label="Building 3")
#plt.plot(df.index[beginning_index:end_index], df["power factor"]["apparent"][beginning_index:end_index], label="Apparent Power")

plt.xlabel("Time")
plt.ylabel("Power")
plt.title("Fridge readings from multiple buildings")
plt.legend()

plt.show()