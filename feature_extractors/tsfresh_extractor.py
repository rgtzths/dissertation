import pandas as pd
import numpy as np
from tsfresh import extract_relevant_features
from tsfresh.feature_extraction.settings import from_columns
from tsfresh.feature_extraction import extract_features



def get_tsfresh_features(dfs, timestep, timewindow, overlap, mains_mean=None, mains_std=None, app_dfs=None, parameters=None):
    X = []
    if app_dfs is not None:
        data = pd.DataFrame(generate_app_timeseries(app_dfs, timewindow, timestep, overlap))
        y =  pd.Series(data[1])
        y.index = data[0]

    n_columns = len(dfs[0].columns.values)

    step = int((timewindow - overlap)*n_columns/timestep)

    window_size = int(timewindow * n_columns /timestep)

    pad = window_size - step
    timeseries_id = 0

    for idx in range(0, len(dfs)):
        
        new_mains = dfs[idx].values.flatten()

        new_mains = np.pad(new_mains, (pad, 0),'constant', constant_values=(0,0))
        


        new_mains = np.concatenate(
                    np.array( [ np.concatenate((new_mains[i : i + window_size].reshape(-1, n_columns), np.full((int(window_size/n_columns), 1), ( timeseries_id + temp_id )) ), axis=1 ) 
                        for temp_id, i in enumerate(range(0, len(new_mains) - window_size + 1, step)) ] ), 
                    axis=0)
                    
        timeseries_id += new_mains[-1][-1]

        X.append(pd.DataFrame(new_mains, columns= [i[0] for i in dfs[idx].columns.values] + ["id"]))

    X = pd.concat(X, axis=0)
    if parameters is None:
        X = extract_relevant_features(X, y, column_id="id", n_jobs=4)
        
        parameters = from_columns(X)
    else:
        X = extract_features(X, kind_to_fc_parameters=parameters, column_id="id", n_jobs=4)

    X = X.values

    if mains_mean is None:
        mains_mean = np.mean(X, axis=0)
        mains_std = np.std(X, axis=0)

    X = (X - mains_mean) / mains_std

    return X, mains_mean, mains_std, parameters

def generate_app_timeseries(dfs, timewindow, timestep, overlap):
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

    timeseries_id = 0
    for idx in range(0, len(dfs)):

        app = dfs[idx].values.flatten()

        app = np.pad(app, (pad, 0),'constant', constant_values=(0,-1))

        [ data.append([ timeseries_id + temp_id, 1]) if app[i+window_size-pad] > 15 else data.append([timeseries_id + temp_id, 0]) for temp_id, i in  enumerate(range(0, len(app) - window_size +1, step)) ]  

        timeseries_id += data[-1][0]
    return np.array(data)