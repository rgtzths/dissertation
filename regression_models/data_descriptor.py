from nilmtk import DataSet

'''
    Data access example
'''
aveiro = DataSet('../../datasets/avEiro_h5/avEiro.h5')

elec = aveiro.buildings[1].elec
print(aveiro.buildings[1])
print("Eletric Appliences of Building 1")
print(elec)

heater = elec[3]

print("Eletric mesurements for the heat pump in building 1")
print(heater.available_columns())

# Load all columns (default) of the heater
df = next(heater.load())
print("Head data stored for heater")
print(df[0:50])


#Load a Single collum
series = next(heater.power_series())
print("Loading a power series")
print(series.head())


#Load a specific column (physical quantity and AC type)

df = next(heater.load(physical_quantity='voltage', ac_type=''))
print("Loading tail of the dataset")
print(df.tail())

# resample to minutely (i.e. with a sample period of 60 secs)
df = next(heater.load(physical_quantity='power', ac_type='apparent', sample_period=60))
print("Loading a resample version of the data with a sampling period of 1 minute")
print(df.head())