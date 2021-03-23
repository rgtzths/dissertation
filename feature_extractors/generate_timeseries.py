import pandas as pd
import numpy as np

def generate_main_timeseries(dfs, is_test, timewindow, timestep):
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
        Time gap covered by each feature vector in min
    timestep : int
        Time between each reading in seconds
    Returns
    -------
    Numpy array of feature vectors
    """
    data = []
    for df in dfs:
        if is_test:
            current_index = 0
            previous_values = []
            while current_index < len(df):

                if len(previous_values) == 0:
                    values = np.zeros(int(timewindow*60/timestep) -1)
                if len(previous_values) != 0: 
                    values = previous_values[1:]
                
                while len(values) < int(timewindow*60/timestep) and current_index < len(df):
                    values.append(df.loc[df.index[current_index]].values)
                    current_index += 1
                if len(values)  == int(timewindow*60/timestep):
                    data.append(values)
                    previous_values = values
            data = np.array(data)

        else:
            values = df.values
            values = values[:int(values.shape[0]/int(timewindow*60/timestep)) * int(timewindow*60/timestep)]
            values = values.reshape(int(values.shape[0]/int(timewindow*60/timestep)) , int(timewindow*60/timestep), len(df.columns.values))

            if len(data) == 0:
                data = values
            else:
                data = np.concatenate((data, values), axis=0)
    
    return data

def generate_appliance_timeseries(dfs, is_classification, timewindow, timestep, column):
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

    #Starting the conversion (goes through all the dataframes)
    for df in dfs:
        values = df[column].values
        values = values[:int(values.shape[0]/int(timewindow*60/timestep)) * int(timewindow*60/timestep)]
        values = values.reshape(int(values.shape[0]/int(timewindow*60/timestep)), int(timewindow*60/timestep))

        if is_classification:
            for feature_vector in values:
                is_postive = False
                for value in feature_vector:
                    if value > 20:
                        is_postive = True
                        data.append(1)
                        break

                if not is_postive:
                    data.append(0)

        else:
            for feature_vector in values:
                data.append(feature_vector[-1])

    return np.array(data)