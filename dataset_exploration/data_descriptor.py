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
aveiro = DataSet('../../datasets/avEiro_h5/avEiro.h5')


print("Getting the dataset metadata.")
pprint(aveiro.metadata)
print("\n\n")

print("Getting the available buildings.")
pprint(aveiro.buildings)
print("\n\n")

print("Getting the metadata of the first building.")
pprint(aveiro.buildings[1].metadata)
print("\n\n")

print("Getting the meters of the first building.")
pprint(aveiro.buildings[1].elec)
print("\n\n")

print("Getting the available collumns for one meter.")
pprint(aveiro.buildings[1].elec['heat pump'].available_columns())
print("\n\n")

print("Loading a single collumn of one meter.")
pprint(next(aveiro.buildings[1].elec['heat pump'].load(physical_quantity="power", ac_type="apparent")))
print("\n\n")

print("Loading the collumns of power (specific physical quantity).")
pprint(next(aveiro.buildings[1].elec['heat pump'].load(physical_quantity="power")))
print("\n\n")

print("Loading the collumns of apparent energy (specific ac type).")
pprint(next(aveiro.buildings[1].elec['heat pump'].load(ac_type="apparent")))
print("\n\n")

print("Loading the collumns of power with a specific resampling.")
pprint(next(aveiro.buildings[1].elec['heat pump'].load(physical_quantity="power", sample_period=60)))
print("\n\n")

print("Getting the submetered propotion of energy measured.")
pprint(aveiro.buildings[1].elec.proportion_of_energy_submetered())
print("\n\n")

print("Getting the total aggregated energy consumed in kWh.")
pprint(aveiro.buildings[1].elec.mains().total_energy())
print("\n\n")

print("Getting the total energy consumed by each submeters in kWh.")
pprint(aveiro.buildings[1].elec.submeters().energy_per_meter())
print("\n\n")

print("Getting the top  k (5) appliances in a building.")
pprint(aveiro.buildings[1].elec.submeters().submeters().select_top_k(k=5))
print("\n\n")


print("Ploting Piechart of energy consumed by appliances")
fraction = aveiro.buildings[1].elec.submeters().fraction_per_meter().dropna()
labels = aveiro.buildings[1].elec.get_labels(fraction.index)
plt.figure(figsize=(10,30))
fraction.plot(kind='pie', labels=labels)
print("\n\n")

print("Ploting the appliances ON state.")
aveiro.buildings[1].elec.plot_when_on(on_power_threshold = 40)
print("\n\n")