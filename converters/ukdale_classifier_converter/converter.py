from os.path import join, isdir, isfile
from os import listdir
import re
from sys import stdout
import pandas as pd
import numpy as np

def convert_ukdale(ukdale_path, output_path):
    """
    Converts the ukdale dataset into a timeseries dataset.
    Parameters
    ----------
    ukdale_path : str
        The root path of the ukdale low_freq dataset.
    output_path : str
        The destination path for the ukdale dataset timeseries.
    timeframe : int
        Time gap covered by each feature vector in min
    timestep : int
        Time between each reading in seconds
    overlap : float
        Amount of overlap between each feacture vector (percentage)
    """

    #Finds all the houses in the dataset
    houses = _find_all_houses(ukdale_path)

    #Starting the conversion (goes through all the houses)
    for house_id in houses:
        print("Loading house ", house_id)

        filenames = []
        #Goes through all the appliances to be converted from that house
        for appliance, meter in house_appliances_mappings["house"+str(house_id)].items():
            print("Converting ", appliance)

            csv_filename = ukdale_path + "house_" + str(house_id) + "/channel_" + meter + ".dat"       
            df = pd.read_csv(csv_filename, sep=" ", header=None)
            df.columns = ["time", "power"]

            #Converts the time column from a unix timestamp to datetime and uses it as index
            df.index = pd.to_datetime(df["time"], unit='s')

            #Drops unnecessary column         
            df = df.drop("time", 1)

            #Sort index and drop duplicates
            df = df.sort_index()
            dups_in_index = df.index.duplicated(keep='first')
            if dups_in_index.any():
                df = df[~dups_in_index]
            
            df.to_csv('{}/house_{}/{}.csv'.format(output_path, house_id, appliance))

        print()

    print("Done converting UKDale for classification!")

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
    dir_names = [p for p in listdir(input_path) if isdir(join(input_path, p))]
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

#Mappings between the houses and the appliances we want to convert
house_appliances_mappings = {
    "house1" : {
        "mains" : "1",
        "washing_machine": "5",
        "dish_washer": "6",
        "fridge": "12",
        "microwave": "13",
        "boiler": "2"
    },
    "house2" : {
        "mains" : "1",
        "washing_machine": "12",
        "dish_washer": "13",
        "kettle": "8",
        "toaster": "16",
        "fridge": "14",
        "microwave": "15",
    },
    "house3" : {
        "mains" : "1",
        "kettle": "2",
    },
    "house5" : {
        "mains" : "1",
        "oven": "20",
        "dish_washer": "22",
        "kettle": "18",
        "toaster":"15",
        "fridge": "19",
        "microwave": "23",
    },
    "house4" : {}
}

filespath = "../../../datasets/ukdale/"
output_path = "../../../datasets/ukdale_classification"

convert_ukdale(filespath, output_path)