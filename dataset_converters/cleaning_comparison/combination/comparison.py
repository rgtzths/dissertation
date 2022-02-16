
import pandas as pd
import numpy as np
import traces
import matplotlib.pyplot as plt

import sys
sys.path.insert(1, "../../")
sys.path.insert(1, "../../../utils")
from sklearn.metrics import mean_squared_error

from data_clean import clean_data

def rmse(app_gt, app_pred):
    return mean_squared_error(app_gt,app_pred)**(.5)

def parse_datetime_ukdale(value):
    return pd.to_datetime(value, format='%Y-%m-%d %H:%M:%S')

def convert_traces(filename, timestep, interpolate):
    """
    Parameters
    ----------
    filename : str
        File used for comparison.
    output_path : str
        The destination path.
    timestep : int
        Amount of seconds between samples.
    interpolate : str
        Either previous or linear (used for filling missing values).
    """

    step = pd.Timedelta(timestep, unit="s")

    ts = traces.TimeSeries.from_csv(
            filename,
            time_column=0,
            time_transform=parse_datetime_ukdale,
            value_column=1,
            value_transform=float,
            default=0,
        )
    if interpolate != "moving_average":
        data = []
        current_time = ts.first_key()

        while current_time <= ts.last_key():
            
            data.append([current_time, ts.get(current_time, interpolate=interpolate)])
            current_time += step

        new_df = pd.DataFrame(data)
        new_df = new_df.set_index(0)
    else:
        new_df = pd.DataFrame(ts.moving_average(timestep, window_size=5,  pandas=True))
        
    return new_df

def clean_dataframe(raw_data, smothing, window):
    """
    Parameters
    ----------
    filename : str
        File used for comparison.
    timestep : int
        Amount of seconds between samples.
    """

    if(smothing == "sma"):
        raw_data["value"] = raw_data["power"].rolling(window=window).mean()

    elif(smothing == "ema"):
        raw_data["value"] = raw_data["power"].ewm(span=window,adjust=False).mean()
    
    elif(smothing == "median"):
        raw_data["value"] = raw_data["power"].rolling(window=window).median()

    return raw_data

def gaussian_noise(noise_df, percentage, mean):
    stds = [1, 1.5, 3, 9, 27]
    random_indices = np.random.randint(0, noise_df.size, int(noise_df.size*percentage))
    examples_per_std = len(random_indices)//len(stds)
    
    for i in range(0,len(stds)):
        noise = np.random.normal(mean, stds[i], noise_df.size)
        for j in range(0, examples_per_std):
            noise_df["power"][random_indices[i*examples_per_std +j]] += noise[random_indices[i*examples_per_std +j]]

    return noise_df

def add_gaps(data, n_gaps=50):
    gaps = [1,3, 5, 10, 20]
    a = range(20 * 10, data.size - 20*30, 20*2)
    random_indices = np.random.choice(a, n_gaps, replace=False)


    for i in range(0,len(gaps)):
        for j in range(0, 10):
            data = data.drop(data.index[ random_indices[i*10 + j] : random_indices[i*10 + j] + gaps[i]])
    
    return data

if __name__ == "__main__":

    input_file = "./original.csv"
    timestep = 6
    algorithms = ["traces", "combination"]
    begining = pd.to_datetime('2014-01-01')
    end = pd.to_datetime('2014-01-02')

    original = pd.read_csv(input_file, sep=",", header=None)
    original.columns = ["time", "power"]
    original.index = pd.to_datetime(original["time"], format='%Y-%m-%d %H:%M:%S')        
    original = original.drop("time", 1)
    begining_index = original.index.get_loc(begining, method="nearest")
    end_index = original.index.get_loc(end, method="nearest")
    original = original[begining_index:end_index]

    results = {}
    processed_file = "processed_file.csv"
    
    for i in range(0,10):
        print("-"*5,"Epoch", i, "-"*5)
        df = original.copy(deep=True)
        df = add_gaps(df)
        df = gaussian_noise(df, 0.2, 0)
        df.to_csv(processed_file, header=None)

        for algorithm in algorithms:
            if algorithm == "traces":
                processed_df = convert_traces(processed_file, timestep, "moving_average")
                processed_df.columns = ["power"]
                error = rmse(processed_df["power"], original["power"][1:])
            else:
                processed_df = convert_traces(processed_file, timestep, "linear")
                processed_df.columns = ["power"]
                processed_df = clean_dataframe(processed_df, "median_filter", 3)
                error = rmse(processed_df["power"], original["power"][1:])

            if algorithm in results:
                results[algorithm].append(error)
            else:
                results[algorithm] = [error]

    for algorithm in results:
        print("Average error in", algorithm, ":", sum(results[algorithm])/len(results[algorithm]) )
        print("Std deviation in", algorithm, ":", np.std(results[algorithm]) )