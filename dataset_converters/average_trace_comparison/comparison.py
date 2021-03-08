from os.path import join, isdir, isfile
from os import listdir
import re
from sys import stdout
import pandas as pd
import numpy as np
import traces
import datetime
import matplotlib.pyplot as plt

def convert_aveiro(filename, output_path, timeframe, timestep):
    """
    Parameters
    ----------
    filename : str
        File used for comparison.
    output_path : str
        The destination path.
    timeframe : int
        Time covered in each frame in min
    """
    
    print("Converting avEiro to Timeseries using average.")
    
    one_second = pd.Timedelta(1, unit="s")
    
    step = pd.Timedelta(timestep, unit="s")

    df = pd.read_csv(input_file)
    df.index = pd.to_datetime(df["time"], unit='ns')
    df.index = df.index.round("s", ambiguous=False)
    df = df.drop("time", 1)
    df = df.drop("name", 1)
    df.columns = ["power"]
    df = df.sort_index()

    dups_in_index = df.index.duplicated(keep='first')
    if dups_in_index.any():
        df = df[~dups_in_index]

    data = []

    current_time = df.index[0]
    current_index = 0
    aprox = 0
    arred = 0
    behind = 0

    while current_index < len(df):

        index_time = df.index[current_index].round("s", ambiguous=False)

        if index_time == current_time or index_time - one_second == current_time or index_time + one_second == current_time:
            data.append([current_time, df["power"][df.index[current_index]]])
            current_index += 1
            current_time += step
            aprox += 1

        elif current_time > index_time:
            if  current_index < len(df) -1:
                next_index = df.index[current_index+1].round("s", ambiguous=False)
                if next_index == current_time or next_index - one_second == current_time:
                    data.append([current_time, df["power"][df.index[current_index+1]]])
                    current_time += step
                    behind += 1
            current_index += 2

        else:
            data.append([current_time, (df["power"][df.index[current_index]] + data[-1][1])/2])
            current_time += step
            arred += 1
        
    new_df = pd.DataFrame(data)
    new_df = new_df.set_index(0)
    new_df.to_csv(output_path+"timeseries.csv", header=False)
    
    print()
    print("Aprox Values: ", aprox)
    print("Arred values: ", arred)
    print("Behind values:", behind)
    print()
    print("Done converting avEiro to Timeseries using average.")

def parse_iso_datetime(value):
    return pd.to_datetime(np.int64(value), unit='ns').round("s")

def convert_aveiro_traces(filename, output_path, timeframe, timestep, interpolate):
    """
    Parameters
    ----------
    filename : str
        File used for comparison.
    output_path : str
        The destination path.
    timeframe : int
        Time covered in each frame in min
    """

    print("Converting avEiro to Timeseries using traces - "+interpolate+".")
    
    one_second = pd.Timedelta(1, unit="s")
    
    step = pd.Timedelta(timestep, unit="s")

    ts = traces.TimeSeries.from_csv(
            filename,
            time_column=1,
            time_transform=parse_iso_datetime,
            value_column=2,
            value_transform=float,
            default=0,
        )

    data = []
    current_time = ts.first_key()

    while current_time < ts.last_key():
        
        data.append([current_time, ts.get(current_time, interpolate=interpolate)])
        current_time += step

    new_df = pd.DataFrame(data)
    new_df = new_df.set_index(0)
    new_df.to_csv(output_path+"traces_"+interpolate+".csv", header=False)
    
    print("Done converting avEiro to Timeseries using traces - "+interpolate+".")


input_file = "../../../datasets/avEiro/house_1/mains/power.csv"
output_path = "../../../datasets/comparison/"

begining = pd.to_datetime('2021-01-09T18:32:00')
end = pd.to_datetime('2021-01-09T18:38:00')

create_csv = False

timeframe = 10
timestep = 2

if create_csv:
    averaged_dates = convert_aveiro(input_file, output_path, timeframe, timestep)
    
    convert_aveiro_traces(input_file, output_path, timeframe, timestep, "previous")
    convert_aveiro_traces(input_file, output_path, timeframe, timestep, "linear")

timeseries = pd.read_csv(output_path+"timeseries.csv", header=None, names=["time", "value"])
timeseries["time"] = pd.to_datetime(timeseries["time"])

traces_previous = pd.read_csv(output_path+"traces_previous.csv", header=None, names=["time", "value"])
traces_previous["time"] = pd.to_datetime(traces_previous["time"])

traces_linear = pd.read_csv(output_path+"traces_linear.csv", header=None, names=["time", "value"])
traces_linear["time"] = pd.to_datetime(traces_linear["time"])

begining_index = 0

while(timeseries["time"][begining_index] != begining):
    begining_index += 1

end_index = begining_index

while(timeseries["time"][end_index] != end):
    end_index += 1

plt.plot(timeseries["time"][begining_index:end_index], timeseries["value"][begining_index:end_index], label="Timeseries")
plt.plot(traces_previous["time"][begining_index:end_index], traces_previous["value"][begining_index:end_index], label="Traces Previous")
plt.plot(traces_linear["time"][begining_index:end_index], traces_linear["value"][begining_index:end_index], label="Traces Linear")

plt.xlabel("Time")
plt.ylabel("Power")
plt.title("Comparison between traces and our method")
plt.legend()

plt.show()