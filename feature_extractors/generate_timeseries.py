import pandas as pd
import numpy as np

def generate_main_timeseries(dfs, is_test, timeframe, timestep, overlap, interpolate):
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
    timeframe : int
        Time gap covered by each feature vector in min
    overlap : float
        Amount of overlap between each feacture vector (percentage)
    timestep : int
        Time between each reading in seconds
    Returns
    -------
    Numpy array of feature vectors
    """
    #Definition of variables to be used during conversion
    columns = list(dfs[0].columns.values)
    one_second = pd.Timedelta(1, unit="s")
    step = pd.Timedelta(timestep, unit="s")
    
    data = []
    #Starting the conversion (goes through all the dataframes)
    for df in dfs:
        #Current time contains the fixed time that is incremented
        #with the step.
        current_time = df.index[0].round("s", ambiguous=False)
        #Current index represents the index of the dataframe.  
        current_index = 0
        
        #Checks if data is to be used during test.
        if is_test:
            #Defines the time step between the beginning of the feature vector and the end, considering the overlap.
            objective_step = step
            #Objective time represents the time where the feature vector is done.
            objective_time = current_time + step
        else:
            objective_step = pd.Timedelta(timeframe*60, unit="s") - pd.Timedelta(timeframe*60*overlap, unit="s")
            objective_time = current_time + objective_step*2
            #Overlap index states the index where the overlap between feature vectors starts
            overlap_index =  int(timeframe*60*len(columns)/timestep - timeframe*60*overlap*len(columns)/timestep)
        
        #Past feature vector is the vector that stores the previous feature vector
        past_feature_vector = []
        #Makes sure that we stop at the end of the df.
        while current_index < len(df):
            #New feacture vector
            feature_vector = []
            #Checks if is test and there is a past vector to fill the new feature vector with the overlap
            if len(past_feature_vector) != 0 and is_test:
                feature_vector = past_feature_vector[1:]
            #Checks if the past fecture vector is empty and that is test to fill the feacture vector with padding.
            elif len(past_feature_vector) == 0 and is_test:
                feature_vector = [0 for i in range(0, int(timeframe*60*len(columns)/timestep -1))]
            #Fills the feacture vector with the overlap intended in case of not being a test and the past feacture vector is not empty.
            elif len(past_feature_vector) != 0:
                feature_vector = past_feature_vector[overlap_index:]

            #Does the cicle until the feacture vector is complete or the end of the dataframe is reached
            while current_time != objective_time and current_index < len(df):
                #Gets the time that corresponds to the current index
                index_time = df.index[current_index].round("s", ambiguous=False)
                #If the index time is equal or +/- 1 second that the objective time.
                #The value in that index is used for the feature vector.
                if index_time == current_time or index_time - one_second == current_time or index_time + one_second == current_time:
                    for c in columns:
                        feature_vector.append(df[c][df.index[current_index]])
                    current_index += 1
                    current_time += step
                #If the index time is behind the current time more than 1 second
                elif current_time > index_time:
                    if  current_index < len(df) -1:
                        # The current index is incremented and we check if the index time is equal or +/- 1 second (same logic as previous if)
                        next_index = df.index[current_index+1].round("s", ambiguous=False)
                        if next_index == current_time or next_index - one_second == current_time:
                            for c in columns:
                                feature_vector.append(df[c][df.index[current_index+1]])
                            current_time += step
                    current_index += 2
                #The last condition indicates that the current time is behind the index time for more than 1 second.
                #In this case the average between the previous feature and the feature in the index time is used for the feature vector.
                else:
                    for c in columns:
                        if interpolate == "average":
                            feature_vector.append((df[c][df.index[current_index]] + feature_vector[-len(columns)])/2)
                        else:
                            feature_vector.append(feature_vector[-len(columns)])
                    current_time += step

            #Checks if the feature vector is the right size (in case the last feature vector hasn't the right size)
            if len(feature_vector) == int(timeframe*60*len(columns)/timestep):
                objective_time += objective_step
                past_feature_vector = feature_vector
                #Add the feature vector to the array.
                data.append(feature_vector)
    
    return np.array(data)

def generate_appliance_timeseries(dfs, is_classification, timeframe, timestep, overlap, column, interpolate):
    """
    Converts an array of dataframes with the format timestamp : value into an
    array of values.

    Parameters
    ----------
    dfs : array of dataframes
        dataframes of format timestamp : value to be used during classification.
    timeframe : int
        Time gap covered by each feature vector in min
    overlap : float
        Amount of overlap between each feacture vector (percentage)
    timestep : int
        Time between each reading in seconds
    column : string
        Name of the column used to extract the values
    Returns
    -------
    Numpy array of feature vectors
    """
    #Definition of variables to be used during conversion
    columns = list(dfs[0].columns.values)
    one_second = pd.Timedelta(1, unit="s")
    step = pd.Timedelta(timestep, unit="s")

    data = []

    #Starting the conversion (goes through all the dataframes)
    for df in dfs:
        #Current time contains the fixed time that is incremented
        #with the step.
        current_time = df.index[0].round("s", ambiguous=False)
        #Current index represents the index of the dataframe.  
        current_index = 0
        #Defines the time step between the beginning of the feature vector and the end, considering the overlap.
        objective_step = pd.Timedelta(timeframe*60, unit="s") - pd.Timedelta(timeframe*60*overlap, unit="s")
        #Objective time represents the time where the feature vector is done.
        objective_time = current_time + objective_step*2

        #Makes sure that we stop at the end of the df.
        while current_index < len(df):
            #Holds the value to be submited when we arrive athe the objective time (t0)
            current_value = 0

            #Does the cicle until we reach the objective time or the end of the dataframe is reached
            while current_time != objective_time and current_index < len(df):
                #Gets the time that corresponds to the current index
                index_time = df.index[current_index].round("s", ambiguous=False)

                #If the index time is equal or +/- 1 second that the objective time.
                #The value in that index is used for the feature vector.
                if index_time == current_time or index_time - one_second == current_time or index_time + one_second == current_time:
                    if is_classification and current_value == 0 and df[column[0]][column[1]][df.index[current_index]] > 10:
                        current_value = 1
                    elif not is_classification:
                        current_value = df[column[0]][column[1]][df.index[current_index]]
                    current_index += 1
                    current_time += step
                #If the index time is behind the current time more than 1 second
                elif current_time > index_time:
                    if  current_index < len(df) -1:
                        next_index = df.index[current_index+1].round("s", ambiguous=False)
                        if next_index == current_time or next_index - one_second == current_time:
                            if is_classification and current_value == 0 and df[column[0]][column[1]][df.index[current_index]] > 10:
                                current_value = 1
                            elif not is_classification:
                                current_value = df[column[0]][column[1]][df.index[current_index]]
                            current_time += step
                    current_index += 2
                #The last condition indicates that the current time is behind the index time for more than 1 second.
                #In this case the average between the previous feature and the feature in the index time is used for the feature vector.
                else:
                    if interpolate == "average":
                        if is_classification and current_value == 0 and (df[column[0]][column[1]][df.index[current_index]] + df[column[0]][column[1]][df.index[current_index -1]])/2  > 10:
                            current_value = 1
                        elif not is_classification:
                            current_value = (df[column[0]][column[1]][df.index[current_index]] + df[column[0]][column[1]][df.index[current_index -1]])/2 
                    current_time += step
            if current_time == objective_time:
                #Add the value to the array.
                data.append(current_value)
            objective_time += objective_step

    return np.array(data)