from os.path import join, isdir, isfile
from os import listdir
import re
from sys import stdout
import pandas as pd
import numpy as np
import traces

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

def parse_iso_datetime(value):
    return pd.to_datetime(np.int64(value), unit='ns').round("s")

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
    
            tss = []
            #If it is mains there are multiple readings otherwise there is only power
            if appliance == "mains":
                #Loads all the readings
                for measure in columns_names:

                    csv_filename = aveiro_path + "house_" + str(house_id) + "/" + str(appliance) + "/" + measure + ".csv"
                    ts = traces.TimeSeries.from_csv(
                        csv_filename,
                        time_column=1,
                        time_transform=parse_iso_datetime,
                        value_column=2,
                        value_transform=float,
                        default=0,
                    )

                    #Store the dataframe into an array
                    tss.append(ts)
            else:
                #Same logic as the presented in the mains processing but using only onde measure.
                measure = "power"
                ts = traces.TimeSeries.from_csv(
                        csv_filename,
                        time_column=1,
                        time_transform=parse_iso_datetime,
                        value_column=2,
                        value_transform=float,
                        default=0,
                    )

                tss.append(ts)

            data = []

            print("\t\tConverting to feature vectors.")
            #Current time contains the fixed time that is incremented
            #with the step.
            current_time = current_time = tss[0].first_key()
            #Objective time represents the time where the feature vector is done.
            objective_time = current_time + objective_step*2
            #Overlap index states the index where the overlap between feature vectors starts
            overlap_index = int(timeframe*60/timestep - timeframe*60*overlap*len(tss)/timestep)
            #Past feature vector is the vector that stores the previous feature vector
            past_feature_vector = []

            #Makes sure that we stop at the end of the df.
            while current_time < tss[0].last_key():
                #New feacture vector
                feature_vector = []

                #When past feacture vector has items, we need to use the overlap.
                if len(past_feature_vector) != 0:
                    feature_vector = past_feature_vector[overlap_index:]

                #Does the cicle until the feacture vector is complete or the end of the dataframe is reached
                while current_time != objective_time and current_time < ts.last_key():
                    for i in range(0, len(tss)):
                        feature_vector.append(tss[i].get(current_time))
                    current_time += step

                #Checks if the feature vector is the right size (in case the last feature vector hasn't the right size)
                if len(feature_vector) == int(timeframe*60*len(tss)/timestep):
                    past_feature_vector = feature_vector
                    #When we are building the mains we add the write the objective time and the feature vector
                    if appliance == "mains":
                        data.append([objective_time]+ feature_vector)

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