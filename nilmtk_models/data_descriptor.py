from nilmtk import DataSet

'''
    Data access example
'''
aveiro = DataSet('../converters/avEiro.h5')

elec = aveiro.buildings[1].elec

print("Eletric Appliences of Building 1")
print(elec)

heater = elec['electric shower heater']

print("Eletric mesurements for the heater in building 1")
print(heater.available_columns())

# Load all columns (default) of the heater
df = next(heater.load())
print("Head data stored for heater")
print(df.head())


#Load a Single collum
series = next(heater.power_series())
print("Loading a single column as a series")
print(series.head())


#Load a specific column (physical quantity and AC type)

df = next(heater.load(physical_quantity='power', ac_type='apparent'))
print("Loading a single column as a data frame")
print(df.tail())

# resample to minutely (i.e. with a sample period of 60 secs)
df = next(heater.load(physical_quantity='power', ac_type='apparent', sample_period=60))
print("Loading a resample version of the data with a sampling period of 1 minute")
print(df.head())