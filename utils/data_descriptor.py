from nilmtk import DataSet
import warnings
warnings.filterwarnings("ignore")
from pprint import pprint
from matplotlib import rcParams
import matplotlib.pyplot as plt
import datetime
from nilmtk import Appliance
import matplotlib.dates as mdates
'''
    Data access example
'''

Appliance.allow_synonyms = False

#dataset = DataSet('../../datasets/avEiro/avEiro.h5')
#dataset = DataSet('../../datasets/ukdale/ukdale.h5')
#dataset.set_window(start=datetime.datetime(2013, 2, 18), end=datetime.datetime(2013, 2, 19))
#dataset = DataSet('../../datasets/ampds2/AMPds2.h5')
dataset = DataSet('../../datasets/iawe/iawe.h5')
#dataset = DataSet('../../datasets/withus/withus.h5')
#dataset = DataSet('../../datasets/refit/refit.h5')
#dataset = DataSet('../../datasets/eco/eco.h5')
#dataset = DataSet('../../datasets/dred/dred.h5')
#dataset = DataSet('../../datasets/combed/combed.h5')
#dataset = DataSet('../../datasets/greend/greend.h5')

#print("Getting the dataset metadata.")
#pprint(dataset.metadata)
#print("\n\n")
#
#print("Getting the available buildings.")
#pprint(dataset.buildings)
#print("\n\n")
#
#print("Getting the metadata of the first building.")
#pprint(dataset.buildings[1].metadata)
#print("\n\n")
#
#
b = 1
print("Getting the meters of the first building.")
pprint(dataset.buildings[b].elec)
print("\n\n")

print(next(dataset.buildings[b].elec["fridge"].load()))

#a = {
#    "kettle" : {
#        "min_off_time" : 0,
#        "min_on_time" : 12,
#        "number_of_activation_padding": 10,
#        "min_on_power" : 2000
#    }
#}

#appliance_name = "kettle"
#activations = dataset.buildings[b].elec[appliance_name].activation_series(
#                        a[appliance_name]["min_off_time"], 
#                        a[appliance_name]["min_on_time"], 
#                        a[appliance_name]["number_of_activation_padding"], 
#                        a[appliance_name]["min_on_power"])
#pprint(activations)


#dataset.set_window(start="2013-06-15 15:13", end="2013-06-15 15:30")
#app_df = next(dataset.buildings[b].elec["fridge"].load(sample_period=1))
#mains_df = next(dataset.buildings[b].elec.mains().load(sample_period=1))

#font = {'family' : 'DejaVu Sans',
#        'weight' : 'bold',
#        'size'   : 20}
#
#plt.rc('font', **font)


#plt.plot(mains_df.index, mains_df["power"]["active"], label="air conditioner active power", linewidth=1.5)
#plt.plot(mains_df.index, mains_df["power"]["apparent"], label="air conditioner reactive power", linewidth=1.5)
#plt.plot(mains_df.index, mains_df["voltage"], label="air conditioner reactive power", linewidth=1.5)
#
#plt.tick_params(
#    axis='x',          # changes apply to the x-axis
#    which='both',      # both major and minor ticks are affected
#    bottom=False,      # ticks along the bottom edge are off
#    top=False,         # ticks along the top edge are off
#    labelbottom=False) # labels along the bottom edge are off
#
#
#plt.subplots_adjust(left=0.07, right=0.96, top=0.94, bottom=0.06)
#
#plt.xlabel("Time", weight='bold')
#plt.ylabel("Power", weight='bold')
#plt.legend()

#plt.show()


#print("Getting the available columns for one meter.")
#pprint(dataset.buildings[1].elec["dish washer"].available_columns())
#print("\n\n")
#
#print("Loading a single collumn of one meter.")
#pprint(next(dataset.buildings[1].elec["fridge"].load(physical_quantity="power", ac_type="active")))
#print("\n\n")
#
#print("Loading the columns of power (specific physical quantity) of charger.")
#pprint(next(dataset.buildings[1].elec["dish washer"].load(physical_quantity="power")))
#print("\n\n")
#
#print("Loading the columns of apparent energy (specific ac type).")
#pprint(next(dataset.buildings[1].elec.mains().load(ac_type="apparent")))
#print("\n\n")

#print("Loading the columns of power with a specific resampling.")
#pprint(next(dataset.buildings[1].elec.mains().load(physical_quantity="power", sample_period=60)))
#print("\n\n")

#print("Getting the submetered propotion of energy measured.")
#pprint(dataset.buildings[20].elec.proportion_of_energy_submetered())
#print("\n\n")
#
#print("Getting the total aggregated energy consumed in kWh.")
#pprint(dataset.buildings[3].elec.mains().total_energy())
#print("\n\n")
#
#
#print("Getting the total energy consumed by each submeters in kWh.")
#
#pprint(dataset.buildings[1].elec.submeters().energy_per_meter())
#
#print("\n\n")
#
#print("Getting the top  k (10) appliances in a building.")
#pprint(dataset.buildings[1].elec.submeters().submeters().select_top_k(k=1))
#print("\n\n")
#
#
#print("Ploting Piechart of energy consumed by appliances")
#fraction = dataset.buildings[1].elec.submeters().fraction_per_meter().dropna()
#labels = dataset.buildings[1].elec.get_labels(fraction.index)
#fraction.plot(kind='pie', labels=labels)
#plt.show()
#print("\n\n")

#print("Ploting the appliances ON state.")
#dataset.buildings[4].elec.plot_when_on(on_power_threshold = 80)
#plt.show()
#print("\n\n")


#print("Obtaining correlation of the appliances")
#correlation_df = elec.pairwise_correlation()
#correlation_df

#print("Obtain appliances by their type")
#print(dataset.buildings[1].elec.select_using_appliances(type=['fridge']))

#print("Mains dropout rate")
#print(dataset.buildings[5].elec.mains().dropout_rate())