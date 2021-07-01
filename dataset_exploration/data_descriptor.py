from nilmtk import DataSet
import warnings
from nilmtk.utils import print_dict
warnings.filterwarnings("ignore")
from pprint import pprint
from matplotlib import rcParams
import matplotlib.pyplot as plt
'''
    Data access example
'''
#dataset = DataSet('../../datasets/avEiro_h5/avEiro.h5')
dataset = DataSet('../../datasets/ukdale/ukdale.h5')
#dataset = DataSet('../../datasets/ampds2/AMPds2.h5')
#dataset = DataSet('../../datasets/iAWE/iawe.h5')

print("Getting the dataset metadata.")
pprint(dataset.metadata)
print("\n\n")

print("Getting the available buildings.")
pprint(dataset.buildings)
print("\n\n")

#print("Getting the metadata of the first building.")
#pprint(dataset.buildings[1].metadata)
#print("\n\n")
#
print("Getting the meters of the first building.")
pprint(dataset.buildings[1].elec)
print("\n\n")

#print("Getting the available columns for one meter.")
#pprint(dataset.buildings[1].elec.mains().available_columns())
#print("\n\n")

#print("Loading a single collumn of one meter.")
#pprint(next(dataset.buildings[1].elec.mains().load(physical_quantity="power", ac_type="apparent")))
#print("\n\n")
#
#print("Loading the columns of power (specific physical quantity) of charger.")
#pprint(next(dataset.buildings[1].elec.mains().load(physical_quantity="power")))
#print("\n\n")
#
#print("Loading the columns of apparent energy (specific ac type).")
#pprint(next(dataset.buildings[1].elec.mains().load(ac_type="apparent")))
#print("\n\n")

#print("Loading the columns of power with a specific resampling.")
#pprint(next(dataset.buildings[1].elec.mains().load(physical_quantity="power", sample_period=60)))
#print("\n\n")

#print("Getting the submetered propotion of energy measured.")
#pprint(dataset.buildings[1].elec.proportion_of_energy_submetered())
#print("\n\n")
#
#print("Getting the total aggregated energy consumed in kWh.")
#pprint(dataset.buildings[1].elec.mains().total_energy())
#print("\n\n")
#
#print("Getting the total energy consumed by each submeters in kWh.")
#pprint(dataset.buildings[1].elec.submeters().energy_per_meter())
#print("\n\n")

print("Getting the top  k (10) appliances in a building.")
pprint(dataset.buildings[1].elec.submeters().submeters().select_top_k(k=10))
print("\n\n")


print("Ploting Piechart of energy consumed by appliances")
fraction = dataset.buildings[1].elec.submeters().fraction_per_meter().dropna()
labels = dataset.buildings[1].elec.get_labels(fraction.index)
fraction.plot(kind='pie', labels=labels)
plt.show()
print("\n\n")

print("Ploting the appliances ON state.")
dataset.buildings[1].elec.plot_when_on(on_power_threshold = 80)
plt.show()
print("\n\n")