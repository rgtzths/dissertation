from os.path import join, isdir
from os import listdir
import re
import pandas as pd
import numpy as np

from nilmtk.measurement import LEVEL_NAMES

import sys
sys.path.insert(1, "../")
from data_clean import clean_data


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

def convert_aveiro(aveiro_path, output_path, columns_names, appliances, timestep, interpolate):
    """
    Converts the avEiro dataset into a timeseries dataset.
    Parameters
    ----------
    aveiro_path : str
        The root path of the avEiro low_freq dataset.
    output_path : str
        The destination path for the avEiro dataset for classification.
    columns_names : list of strings
        Names of the columns used for the aggregated power.
    appliances : list of strings
        Names of the appliances to be converted.
    """

    #Finds all the houses in the dataset
    houses = _find_all_houses(aveiro_path)

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
                for measure in column_mapping:

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

            df = clean_data(df, timestep, interpolate)
            
            df.columns = pd.MultiIndex.from_tuples([column_mapping[x] for x in df.columns])
            df.columns.set_names(LEVEL_NAMES, inplace=True)
            
            df.to_csv('{}/house_{}/{}.csv'.format(output_path, house_id, appliance))
        print()

    print("Done converting avEiro to Timeseries!")

filespath = "../../../../datasets/avEiro/"
output_path = "../../../../datasets/avEiro_classification"

column_mapping = {
    "power" : ("power", "apparent"),
    "vrms" : ("voltage", "")
}

appliances = [
    "mains",
    "heatpump",
    "carcharger"
]

timestep = 2

interpolate = "previous"

convert_aveiro(filespath, output_path, column_mapping, appliances, timestep, interpolate)