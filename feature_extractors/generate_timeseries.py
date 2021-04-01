import pandas as pd
import numpy as np

def generate_main_timeseries(dfs, timewindow, timestep, overlap):
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

    overlap_index = int((timewindow-overlap)*n_columns/timestep)

    step = int((timewindow - overlap)/timestep)

    for df in dfs:
        
        current_index = 0

        if overlap != 0:
            values = np.zeros(overlap_index)
        else:
            values = np.zeros(0)

        while current_index + step < len(df):

            values = np.append(values, df.loc[df.index[current_index:current_index + step ].values].values)

            if len(values) != int(timewindow*n_columns/timestep):
                raise Exception("Invalid length of values ", len(values), int(timewindow*n_columns/timestep))

            data.append(values)

            values = values[overlap_index:]
            
            current_index += step

    data = np.array(data)
    return data

def generate_appliance_timeseries(dfs, is_classification, timewindow, timestep, column, overlap):
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
    column : string
        Name of the column used to extract the values
    Returns
    -------
    Numpy array of feature vectors
    """

    data = []
    step = int((timewindow-overlap)/timestep)

    #Starting the conversion (goes through all the dataframes)
    for df in dfs:
   
        current_index = 0

        while current_index + step < len(df):

            value = df.loc[df.index[current_index + step ], column]
                
            if is_classification:
                
                data.append(1) if value > 20 else data.append(0)
                
            else:
                data.append(value)
            
            current_index += step    

    return np.array(data)