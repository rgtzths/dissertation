import numpy as np
import pandas as pd


def load_data(dataset_folder, appliance, houses):
    #Array that contains the aggragate readings in dataframes
    aggregated_readings = []

    #Dictionary that contains the readings from the appliances
    # With format [ df-house1, df-house2... ]
    app_readings = []

    #Goes through all the houses
    for house in houses:
        #Loads the mains files
        df = pd.read_csv(dataset_folder+house+"/mains.csv", sep=',')
        #Converts the time column from a unix timestamp to datetime and uses it as index
        df.index = pd.to_datetime(df["time"])
        df.index = df.index.round("s", ambiguous=False)
        #Drops unnecessary columns
        df = df.drop("time", 1)
        df = df.sort_index()
        dups_in_index = df.index.duplicated(keep='first')
        if dups_in_index.any():
            df = df[~dups_in_index]
        #appends that dataframe to an array
        aggregated_readings.append(df)

        #Loads the appliance to be used from that house using a similar logic.
        df = pd.read_csv(dataset_folder+house+ "/"+ appliance + ".csv", sep=',')
        df.index = pd.to_datetime(df["time"])
        df.index = df.index.round("s", ambiguous=False)
        #Drops unnecessary columns
        df = df.drop("time", 1)
        df = df.sort_index()
        dups_in_index = df.index.duplicated(keep='first')
        if dups_in_index.any():
            df = df[~dups_in_index]
        #Stores the dataframe in a dictionary
        app_readings.append(df)
    
    #Goes through all the houses.
    for i, house in enumerate(houses.keys()):
        if houses[house]["beginning"] is None:
            #Intersects the intervals between the mains data and the appliance data.
            if aggregated_readings[i].index[0] < app_readings[i].index[0]:
                beginning = app_readings[i].index[0]
            else: 
                beginning = aggregated_readings[i].index[0]
        else:
            if aggregated_readings[i].index[0] > houses[house]["beginning"] and app_readings[i].index[0] > houses[house]["beginning"]:
                raise Exception("Beginning Time before the beginning of the house dataset, use None instead.")
            elif aggregated_readings[i].index[-1] < houses[house]["beginning"] and app_readings[i].index[-1] < houses[house]["beginning"]:
                raise Exception("Beginning Time after the end of the house dataset, use None instead.")
            else:
                beginning = houses[house]["beginning"]

        if houses[house]["end"] is None:
            if aggregated_readings[i].index[-1] < app_readings[i].index[-1]:
                end = aggregated_readings[i].index[-1]
            else:
                end = app_readings[i].index[-1]
        else:
            if aggregated_readings[i].index[-1] < houses[house]["end"] and app_readings[i].index[-1] < houses[house]["end"]:
                raise Exception("End Time after the end of the house dataset, use None instead.")
            elif aggregated_readings[i].index[0] > houses[house]["end"] and app_readings[i].index[0] > houses[house]["end"]:
                raise Exception("End Time before the beginning of the house dataset, use None instead.")
            else:
                end = houses[house]["end"]
        if beginning > end:
                raise Exception("Beginning time is after the end time. Invalid time interval.")

        beginning_index_x = aggregated_readings[i].index.get_loc(beginning, method="nearest")
        end_index_x = aggregated_readings[i].index.get_loc(end, method="nearest")

        beginning_index_y = app_readings[i].index.get_loc(beginning, method="nearest")
        end_index_y = app_readings[i].index.get_loc(end, method="nearest")

        #Updates the dfs to only use the intersection.
        aggregated_readings[i] = aggregated_readings[i][beginning_index_x: end_index_x]
        app_readings[i] = app_readings[i][beginning_index_y: end_index_y]

    return (aggregated_readings, app_readings)

def generate_main_timeseries(dfs, timeframe, timestep, overlap, interpolate):
    """
    Converts an array of dataframes with the format timestamp : value into an
    array of feature vectors.

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
    interpolate : str
        Defines the interpolation method for missing values 
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
        #Timestep between windows (considering the overlap)
        objective_step = pd.Timedelta(timeframe*60, unit="s") - pd.Timedelta(timeframe*60*overlap, unit="s")
        #Time corresponding to the end of the window
        objective_time = current_time + objective_step*2
        #Overlap index states the index where the overlap between feature vectors starts
        overlap_index =  int(timeframe*60*len(columns)/timestep - timeframe*60*overlap*len(columns)/timestep)
        
        #Past feature vector is the vector that stores the previous feature vector
        past_feature_vector = []
        
        #Makes sure that we stop at the end of the df.
        while current_index < len(df):
            #New feacture vector
            feature_vector = []

            #Fills the feacture vector with the overlap intended in case of the past feacture vector is not empty.
            if len(past_feature_vector) != 0:
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


def generate_appliance_timeseries(dfs, timeframe, timestep, overlap, interpolate):
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
        #Past feature vector is the vector that stores the previous feature vector
        #Overlap index states the index where the overlap between feature vectors starts
        overlap_index =  int(timeframe*60/timestep - timeframe*60*overlap*len(columns)/timestep)
        
        past_feature_vector = []
        #Makes sure that we stop at the end of the df.
        while current_index < len(df):
            #New feacture vector
            feature_vector = []

            #Fills the feacture vector with the overlap intended in case of the past feacture vector is not empty.
            if len(past_feature_vector) != 0:
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
                added = False
                for i in feature_vector:
                    if i > 10:
                        data.append([1])
                        added = True
                        break
                if not added:
                    data.append([0])

    return np.array(data)




