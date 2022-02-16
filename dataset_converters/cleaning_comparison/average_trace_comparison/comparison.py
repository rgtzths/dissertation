
import pandas as pd
import numpy as np
import traces

import sys
sys.path.insert(1, "../../")
sys.path.insert(1, "../../../utils")
from sklearn.metrics import mean_squared_error

from data_clean import clean_data

def rmse(app_gt, app_pred):
    return mean_squared_error(app_gt,app_pred)**(.5)

def parse_datetime_ukdale(value):
    return pd.to_datetime(value, format='%Y-%m-%d %H:%M:%S')

def convert_average(filename, timestep):

    convert_df = pd.read_csv(filename, sep=",", header=None)
    convert_df.columns = ["time", "power"]

    #Converts the time column from a unix timestamp to datetime and uses it as index
    convert_df.index = pd.to_datetime(convert_df["time"], format='%Y-%m-%d %H:%M:%S')

    #Drops unnecessary column         
    convert_df = convert_df.drop("time", 1)

    convert_df = clean_data(convert_df, timestep, "average")

    return convert_df

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

def add_gaps(gap_size, data, n_gaps=50):
    a = range(gap_size * 10, data.size - gap_size*30, gap_size*2)
    random_indices = np.random.choice(a, n_gaps, replace=False)

    for i in random_indices:
        data = data.drop(data.index[i:i+gap_size])
    
    return data

if __name__ == "__main__":

    input_file = "./original.csv"
    timestep = 6
    gaps = [1,3, 5, 10, 20]
    algorithms = ["average", "previous", "linear"]
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
        for gap in gaps:
            df = original.copy(deep=True)
            df = add_gaps(gap, df)
            df.to_csv(processed_file, header=None)

            for algorithm in algorithms:
                if algorithm == "average":
                    processed_df = convert_average(processed_file, timestep)
                    error = rmse(processed_df["power"], original["power"])
                else:
                    processed_df = convert_traces(processed_file, timestep, algorithm)
                    processed_df.columns = ["power"]
                    error = rmse(processed_df["power"], original["power"][1:])

                if gap in results:
                    if algorithm in results[gap]:
                        results[gap][algorithm].append(error)
                    else:
                        results[gap][algorithm] = [error]
                else:
                    results[gap] = { algorithm : [error]}

                
    for gap in results:
        print("-"*5,"Using gap:", gap, "-"*5)
        for algorithm in results[gap]:
            print("Average error in", algorithm, ":", sum(results[gap][algorithm])/len(results[gap][algorithm]) )