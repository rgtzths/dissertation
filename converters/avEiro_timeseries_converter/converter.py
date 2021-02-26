from os.path import join, isdir, isfile
from os import listdir
import re
from sys import stdout
import pandas as pd
import numpy as np

def _find_all_houses(input_path):
    """
    Searches for folders in the input path with the prefix 'house_' 
    and returns a list of numbers.

    Parameters
    ----------
    input_path : string
        path to the folder to search for the houses
    Returns
    -------
    list of integers (house instances)
    """
    #Get the directories complete name
    dir_names = [p for p in listdir(input_path) if isdir(join(input_path, p))]
    #Get the numbers of the houses and return them.
    return _matching_ints(dir_names, r'^house_(\d)$')

def _matching_ints(strings, regex):
    """
    Uses regular expression to select and then extract an integer from
    strings.

    Parameters
    ----------
    strings : list of strings
    regex : string
        Regular Expression.  Including one group.  This group is used to
        extract the integer from each string.

    Returns
    -------
    list of ints
    """
    ints = []
    p = re.compile(regex)
    for string in strings:
        m = p.match(string)
        if m:
            integer = int(m.group(1))
            ints.append(integer)
    ints.sort()
    return ints

def convert_aveiro(aveiro_path, output_path, timeframe, timestep, overlap, columns_names, appliances):
    """
    Converts the avEiro dataset into a timeseries dataset.
    Parameters
    ----------
    aveiro_path : str
        The root path of the avEiro low_freq dataset.
    output_path : str
        The destination path for the avEiro dataset timeseries.
    timeframe : int
        Time gap covered by each feature vector in min
    timestep : int
        Time between each reading in seconds
    overlap : float
        Amount of overlap between each feacture vector (percentage)
    columns_names : list of strings
        Names of the columns used for the aggregated power.
    appliances : list of strings
        Names of the appliances to be converted.
    """

    #Finds all the houses in the dataset
    houses = _find_all_houses(aveiro_path)

    #Definition of variables to be used during conversion
    one_second = pd.Timedelta(1, unit="s")
    objective_step = pd.Timedelta(timeframe*60, unit="s") - pd.Timedelta(timeframe*60*overlap, unit="s")
    step = pd.Timedelta(timestep, unit="s")

    #Starting the conversion (goes through all the houses)
    for house_id in houses:
        print("Loading house ", house_id)

        filenames = []
        #Goes through all the appliances
        for appliance in appliances:
            print("\tConverting ", appliance)
    
            dfs = []
            #If it is mains there are multiple readings otherwise there is only power
            if appliance == "mains":
                #Loads all the readings
                for measure in columns_names:

                    csv_filename = aveiro_path + "house_" + str(house_id) + "/" + str(appliance) + "/" + measure + ".csv"
                    df = pd.read_csv(csv_filename)
                    #Converts the time column from a unix timestamp to datetime and uses it as index
                    df.index = pd.to_datetime(df["time"], unit='ns')
                    df.index = df.index.round("s", ambiguous=False)
                    #Drops unnecessary columns
                    df = df.drop("time", 1)
                    df = df.drop("name", 1)
                    #Labels the value column with the appliance name
                    df.columns = [measure]
                    #Sort index and drop duplicates
                    df = df.sort_index()
                    dups_in_index = df.index.duplicated(keep='first')
                    if dups_in_index.any():
                        df = df[~dups_in_index]

                    #Store the dataframe into an array
                    dfs.append(df)
            else:
                #Same logic as the presented in the mains processing but using only onde measure.
                measure = "power"
                csv_filename = aveiro_path + "house_" + str(house_id) + "/" + str(appliance) + "/" + measure + ".csv"
                df = pd.read_csv(csv_filename)
                df.index = pd.to_datetime(df["time"], unit='ns')
                df.index = df.index.round("s", ambiguous=False)
                df = df.drop("time", 1)
                df = df.drop("name", 1)
                df.columns = [measure]
                df = df.sort_index()
                dups_in_index = df.index.duplicated(keep='first')
                if dups_in_index.any():
                    df = df[~dups_in_index]
                dfs.append(df)

            #Concatenate the multiple dataframes ( only relevant when multiple measures are present)
            df = pd.concat(dfs, axis=1)

            data = []
            columns = list(df.columns.values)
            #When there are multiple readings present
            #Sometimes the index don't match so
            #We need to find those cases and substitute the Nan values of the collumns with the
            #average between readings
            if len(columns) > 1:
                print("\t\tReplacing Nans")
                for c in columns:
                    for i in range(0, len(df[c])):
                        if np.isnan(df[c][i]):
                            for j in range(i+1, len(df[c])):
                                if not np.isnan(df[c][j]):
                                    df[c][i] = (df[c][i-1] + df[c][j])/2
                                    break
                        if np.isnan(df[c][i]):
                            for j in range(i, len(df[c])):
                                df[c][j] = df[c][j-1]
                            break

            print("\t\tConverting to feature vectors.")
            #Current time contains the fixed time that is incremented
            #with the step.
            current_time = df.index[0]
            #Current index represents the index of the dataframe.  
            current_index = 0
            #Objective time represents the time where the feature vector is done.
            objective_time = current_time + objective_step*2
            #Overlap index states the index where the overlap between feature vectors starts
            overlap_index = int(timeframe*60/timestep - timeframe*60*overlap*len(columns)/timestep)
            #Past feature vector is the vector that stores the previous feature vector
            past_feature_vector = []
            #Variables for information
            aprox = 0
            arred = 0
            behind = 0

            #Makes sure that we stop at the end of the df.
            while current_index < len(df):
                #New feacture vector
                feature_vector = []

                #When past feacture vector has items, we need to use the overlap.
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
                        aprox += 1
                    #If the index time is behind the current time more than 1 second
                    elif current_time > index_time:
                        if  current_index < len(df) -1:
                            # The current index is incremented and we check if the index time is equal or +/- 1 second (same logic as previous if)
                            next_index = df.index[current_index+1].round("s", ambiguous=False)
                            if next_index == current_time or next_index - one_second == current_time:
                                for c in columns:
                                    feature_vector.append(df[c][df.index[current_index+1]])
                                current_time += step
                                behind += 1
                        current_index += 2
                    #The last condition indicates that the current time is behind the index time for more than 1 second.
                    #In this case the average between the previous feature and the feature in the index time is used for the feature vector.
                    else:
                        for c in columns:
                            feature_vector.append((df[c][df.index[current_index]] + feature_vector[-len(columns)])/2)
                        current_time += step
                        arred += 1
                #Checks if the feature vector is the right size (in case the last feature vector hasn't the right size)
                if len(feature_vector) == int(timeframe*60*len(columns)/timestep):
                    
                    past_feature_vector = feature_vector
                    #When we are building the mains we add the write the objective time and the feature vector
                    if appliance == "mains":
                        data.append([objective_time]+ feature_vector )
                    #Whe we are building the appliances we check if any of the readings is bigger that 10 an
                    #if so we write 1 (device on) else we write 0 (device off) and the objective time.
                    else:
                        added = False
                        for i in feature_vector:
                            if i > 10:
                                data.append([objective_time, 1])
                                added = True
                                break
                        if not added:
                            data.append([objective_time, 0])

                    objective_time += objective_step
            
            new_df = pd.DataFrame(data)
            new_df = new_df.set_index(0)
            new_df.to_csv('{}/house_{}/{}.csv'.format(output_path, house_id, appliance), header=False)
            
            print()
            print("\tAprox Values: ", aprox)
            print("\tArred values: ", arred)
            print("\tBehind values:", behind)
            print()

        print()

    print("Done converting avEiro to Timeseries!")

filespath = "../../../datasets/avEiro/"
output_path = "../../../datasets/avEiro_timeseries"

columns_names = [
    "power",
    "vrms"
]

appliances = [
    "mains",
    "heatpump",
    "carcharger"
]


timeframe = 10
timestep = 2
overlap = 0.5

convert_aveiro(filespath, output_path, timeframe, timestep, overlap, columns_names, appliances)