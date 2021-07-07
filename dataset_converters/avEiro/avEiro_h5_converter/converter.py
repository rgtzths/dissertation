
from nilmtk.utils import get_datastore
from nilmtk.utils import check_directory_exists
from nilmtk.datastore import Key
from nilmtk.measurement import LEVEL_NAMES
from nilm_metadata import save_yaml_to_datastore

from os.path import join, isdir, join
from os import listdir
import re
import pandas as pd
import numpy as np

import sys
sys.path.insert(1, "../../")
sys.path.insert(1, "../../../utils")

from data_clean import clean_data
import utils

def convert_aveiro(aveiro_path, output_filename, timestep, interpolate):
    """
    Converts the avEiro dataset into a h5 dataset ready to be used in nilmtk.
    Parameters
    ----------
    aveiro_path : str
        The root path of the avEiro low_freq dataset.
    output_filename : str
        The destination filename (including path and suffix).
    """

    # Open DataStore
    base_path = ""
    for directory in  output_filename.split("/")[0:-1]:
        base_path += directory + "/"
    utils.create_path( base_path)
    store = get_datastore(output_filename, "HDF", mode='w')

    check_directory_exists(aveiro_path)

    #Gets all houses
    houses = _find_all_houses(aveiro_path)

    #Starts the conversion
    for house_id in houses:
        print("Loading house ", house_id)
        
        for appliance, meter in appliance_meter_mapping.items():
            print("Loading ", appliance)

            #Generate a Key with the building id (1,2,3...) and the meter obtained in the appliance_meter_mapping.
            key = Key(building=house_id, meter=meter)
            
            #When using mains the are multiple measures used
            if appliance == "mains":
                dfs = []
                #Loads the measures and places the dataframes in an array.
                for measure in column_mapping.keys():

                    csv_filename = aveiro_path + "house_" + str(house_id) + "/" + str(appliance) + "/" + measure + ".csv"
                    df = pd.read_csv(csv_filename)
                    #Converts the time column from a unix timestamp to datetime and uses it as index
                    df.index = pd.to_datetime(df["time"], unit='ns')
                    df.index = df.index.round("s", ambiguous=False)
                    #Drops unnecessary columns
                    del df["time"]
                    del df["name"]
                    #Labels the value column with the appliance name
                    df.columns = [measure]
                    #Sort index and drop duplicates
                    df = df.sort_index()
                    dups_in_index = df.index.duplicated(keep='first')
                    if dups_in_index.any():
                        df = df[~dups_in_index]
                    
                    #Store the dataframe into an array
                    dfs.append(df)

                beginning = dfs[0].index[0]
                end = dfs[0].index[-1]
                for df in dfs:
                    if beginning < df.index[0]:
                        beginning = df.index[0]
                    if end > df.index[-1]:
                        end = df.index[-1]
                for i in range(0, len(dfs)):
                    dfs[i] = dfs[i][ dfs[i].index.get_loc(beginning, method="nearest"): dfs[i].index.get_loc(end, method="nearest")]

                #Concatenate the multiple dataframes ( only relevant when multiple measures are present)
                total = pd.concat(dfs, axis=1)

                total = clean_data(total, timestep, interpolate)

                total.columns = pd.MultiIndex.from_tuples([column_mapping[c] for c in total.columns.values], names=LEVEL_NAMES)
                #Store the dataframe in h5
                store.put(str(key), total)
                
            else:
                measure = "power"
                #Same login as in mains but using only one measure.
                csv_filename = aveiro_path + "house_" + str(house_id) + "/" + str(appliance) + "/power.csv"
                df = pd.read_csv(csv_filename)
                df.index = pd.to_datetime(df["time"], unit='ns')
                df.index = df.index.round("s", ambiguous=False)
                del df["name"]
                del df["time"]
                df.columns = [measure]
                df = df.sort_index()
                dups_in_index = df.index.duplicated(keep='first')
                if dups_in_index.any():
                    df = df[~dups_in_index]
                df = clean_data(df, timestep, interpolate)

                df.columns = pd.MultiIndex.from_tuples([column_mapping["power"]], names=LEVEL_NAMES)
                store.put(str(key), df)
        print()

    # Add metadata
    save_yaml_to_datastore(metada_path, store)

    store.close()
    print("Done converting avEiro to HDF5!")
    
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

#Mapping between the names on the dataset and the names used in nilmtk
column_mapping = {
    "power" : ("power", "apparent"),
    "vrms" : ("voltage", "")
}

#Mapping between the names on the dataset and the indexes used in the nilmtk
appliance_meter_mapping = {
    "mains" : 1,
    "heatpump" : 2,
    "carcharger" : 3
}

aveiro_path = "../../../../datasets/avEiro/"
metada_path = aveiro_path + "metadata"
output_filename = "../../../../datasets/avEiro_h5/avEiro.h5"

interpolate = "previous"
timestep = 2

convert_aveiro(aveiro_path, output_filename, timestep, interpolate)