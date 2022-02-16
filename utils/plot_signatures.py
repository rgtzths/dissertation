from nilmtk import DataSet
from matplotlib import pyplot as plt

font = {'family' : 'monospace',
        'size'   : 14,
        'weight': 'bold'
        }
plt.rc('font', **font)



def plot_signature(dataset_location, building, beggining, end):
    
    dataset = DataSet(dataset_location)

    dataset.set_window(start=beggining,end=end)

    car_df = next(dataset.buildings[building].elec["charger"].load())
    mains_df = next(dataset.buildings[building].elec.mains().load())
    heatpump_df = next(dataset.buildings[building].elec["heat pump"].load())
    #plt.plot(car_df.index, car_df["power"]["apparent"], label="Car charger")
    plt.plot(mains_df.index, mains_df["power"]["apparent"], label="Aggregated Power")
    #plt.plot(heatpump_df.index, heatpump_df["power"]["apparent"], label="Heat pump")

    plt.xlabel("Time")
    plt.ylabel("Power")
    plt.legend()
    #plt.title("Mains and " + appliance + " energy consumption from house" + str(building))
    plt.tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False)
    plt.subplots_adjust(left=0.1, right=0.99, top=0.99, bottom=0.06)
    plt.show()
    

location = '../../datasets/avEiro/avEiro.h5'
start ="2020-12-02T11:00"
end="2020-12-02T14:30"

plot_signature(location, 1, start, end)