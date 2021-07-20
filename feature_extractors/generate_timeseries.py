import pandas as pd
import numpy as np

def generate_main_timeseries(dfs, timewindow, timestep, overlap, mains_mean=None, mains_std=None):
    """
    Converts an array of dataframes with the format timestamp : value into an
    array of feature vectors.

    Parameters
    ----------
    dfs : array of dataframes
        dataframes of format timestamp : value to be used during classification.
    is_test : boolean
        Defines if the data is to be used during training or testing
        If true the overlap is ignored and the overlap becomes equal to the lenght of the
        feature vector -2.
    timewindow : int
        Time gap covered by each feature vector in seconds
    timestep : int
        Time between each reading in seconds
    overlap : int
        Time overlaping between each reading in seconds
    Returns
    -------
    Numpy array of feature vectors
    """
    data = []

    n_columns = len(dfs[0].columns.values)
    step = int((timewindow - overlap)*n_columns/timestep)

    window_size = int(timewindow * n_columns /timestep)

    pad = window_size - step

    if mains_mean is None:
        l = np.array(pd.concat(dfs,axis=0))
        mains_mean = np.mean(l, axis=0)
        mains_std = np.std(l, axis=0)

    for df in dfs:
        
        new_mains = df.values

        new_mains = (new_mains - mains_mean) / mains_std

        new_mains = new_mains.flatten()

        new_mains = np.pad(new_mains, (pad, 0),'constant', constant_values=(0,0))

        new_mains = np.array([ new_mains[i : i + window_size] for i in range(0, len(new_mains) - window_size + 1, step)])

        data.append(pd.DataFrame(new_mains))

    data = pd.concat(data, axis=0)

    return (data.values.reshape((-1, int(window_size/n_columns), n_columns)) , mains_mean, mains_std)

def generate_appliance_timeseries(dfs, is_classification, timewindow, timestep, overlap, app_mean=None, app_std=None):
    """
    Converts an array of dataframes with the format timestamp : value into an
    array of values.

    Parameters
    ----------
    dfs : array of dataframes
        dataframes of format timestamp : value to be used during classification.
    timewindow : int
        Time gap covered by each feature vector in min
    timestep : int
        Time between each reading in seconds
    Returns
    -------
    Numpy array of feature vectors
    """

    data = []
    step = int((timewindow-overlap)/timestep)

    window_size = int(timewindow/timestep)
    
    pad = window_size - step

    if app_mean is None:
        l = np.array(pd.concat(dfs,axis=0))
        app_mean = np.mean(l, axis=0)
        app_std = np.std(l, axis=0)

    #Starting the conversion (goes through all the dataframes)
    for df in dfs:
        app = df.values.flatten()
        app = np.pad(app, (pad, 0),'constant', constant_values=(0,-1))

        if is_classification:
            [data.append([0, 1]) if app[i+window_size-1] > 80 else data.append([1, 0]) for i in range(0, len(app) - window_size +1, step) ]    
        else:
            [data.append(app[i+window_size -1]) for i in range(0, len(app) - window_size +1, step)]

    return (np.array(data) - app_mean)/app_std , app_mean, app_std