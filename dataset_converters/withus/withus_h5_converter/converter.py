
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
import math

import sys
sys.path.insert(1, "../../")
sys.path.insert(1, "../../../utils")

from data_clean import clean_data
import utils

def convert_withus(withus_path, output_filename):
    """
    Converts the withus dataset into a h5 dataset ready to be used in nilmtk.
    Parameters
    ----------
    withus_path : str
        The root path of the withus low_freq dataset.
    output_filename : str
        The destination filename (including path and suffix).
    """

    # Open DataStore
    base_path = ""
    for directory in  output_filename.split("/")[0:-1]:
        base_path += directory + "/"

    utils.create_path( base_path)

    store = get_datastore(output_filename, "HDF", mode='w')

    check_directory_exists(withus_path)

    #Gets all houses
    houses = _find_all_houses(withus_path)

    #Starts the conversion
    for house_id in houses:
        print("Loading house", house_id)
        
        for appliance, meter in appliance_meter_mapping[house_id].items():
            print("Loading", appliance)

            #Generate a Key with the building id (1,2,3...) and the meter obtained in the appliance_meter_mapping.
            key = Key(building=house_mapping[house_id], meter=meter)
            
            if appliance == "mains":
                df = []

                csv_filename = withus_path + "house_" + house_id + "/" + str(appliance) + ".csv"
                df = pd.read_csv(csv_filename, header=[0], index_col=0)
                #Converts the time column from a unix timestamp to datetime and uses it as index
                df.index = pd.to_datetime(df.index)

                #Calculate apparent power, apparent cummultative energy and unify the reactive power and cumultative energy
                df = preprocess_mains(df)
                print(df)
                #appends that dataframe to an array
                for c in df.columns:
                    if c not in column_mapping:
                        del df[c]

                #Labels the value column with the appliance name
                df.columns = pd.MultiIndex.from_tuples([column_mapping[c] for c in df.columns.values], names=LEVEL_NAMES)

                #Sort index and drop duplicates
                df = df.sort_index()
                print(df)
                #Store the dataframe in h5
                store.put(str(key), df)
                
            else:
                measure = "power_plus"

                #Same login as in mains but using only one measure.
                csv_filename = withus_path + "house_" + house_id + "/" + str(appliance) + ".csv"

                df = pd.read_csv(csv_filename, header=[0], index_col=0)
                #Converts the time column from a unix timestamp to datetime and uses it as index
                df.index = pd.to_datetime(df.index)

                #appends that dataframe to an array
                for c in df.columns:
                    if c != measure:
                        del df[c]

                print(df)
                #Labels the value column with the appliance name
                df.columns = pd.MultiIndex.from_tuples([column_mapping[measure]], names=LEVEL_NAMES)

                #Sort index and drop duplicates
                df = df.sort_index()
                print(df)
                #Store the dataframe in h5
                store.put(str(key), df)
        print()

    # Add metadata
    save_yaml_to_datastore(metada_path, store)

    store.close()
    print("Done converting withus to HDF5!")
    
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
    dir_names = [p.split("_")[1] for p in listdir(input_path) if isdir(join(input_path, p)) and "house_" in p ]
    return dir_names


def preprocess_mains(df):
    power_apparent = []
    power_reactive = []
    energy_apparent = []
    energy_reactive = []

    for i in df.index:
        if df["power_factor"][i] <= 90:
            power_apparent.append(math.sqrt(df["power_plus"][i]**2 + df["reactive_power1"][i]**2 ))
            power_reactive.append(df["reactive_power1"][i])
            energy_apparent.append(math.sqrt(df["energy_aplus"][i]**2 + df["reactive_energy1"][i]**2 ))
            energy_reactive.append(df["reactive_energy1"][i])
        elif df["power_factor"][i] <=180:
            power_apparent.append(math.sqrt(df["power_plus"][i]**2 + df["reactive_power2"][i]**2 ))
            power_reactive.append(df["reactive_power2"][i])
            energy_apparent.append(math.sqrt(df["energy_aplus"][i]**2 + df["reactive_energy2"][i]**2 ))
            energy_reactive.append(df["reactive_energy2"][i])
        elif df["power_factor"][i] <= 270:
            power_apparent.append(math.sqrt(df["power_plus"][i]**2 + df["reactive_power3"][i]**2 ))
            power_reactive.append(df["reactive_power3"][i])
            energy_apparent.append(math.sqrt(df["energy_aplus"][i]**2 + df["reactive_energy3"][i]**2 ))
            energy_reactive.append(df["reactive_energy3"][i])
        else:
            power_apparent.append(math.sqrt(df["power_plus"][i]**2 + df["reactive_power4"][i]**2 ))
            power_reactive.append(df["reactive_power4"][i])
            energy_apparent.append(math.sqrt(df["energy_aplus"][i]**2 + df["reactive_energy4"][i]**2 ))
            energy_reactive.append(df["reactive_energy4"][i])
    
    del df["reactive_power1"]
    del df["reactive_power2"]
    del df["reactive_power3"]
    del df["reactive_power4"]
    del df["reactive_energy1"]
    del df["reactive_energy2"]
    del df["reactive_energy3"]
    del df["reactive_energy4"]

    df["power_apparent"] = power_apparent
    df["reactive_power"] = power_reactive
    df["apparent_energy"] = energy_apparent
    df["reactive_energy"] = energy_reactive

    return df

#Mapping between the names on the dataset and the names used in nilmtk
column_mapping = {
    "power_plus" : ("power", "active"),
    "reactive_power" : ("power", "reactive"),
    "power_apparent" : ("power", "apparent"),
    "power_factor": ("power factor", "active"),
    "frequency" : ("frequency", "active"),
    "current" : ("current", "active"),
    "voltage" : ("voltage", "active"),
    "energy_aplus" : ("cumulative energy", "active"),
    "reactive_energy" : ("cumulative energy", "reactive"),
    "apparent_energy" : ("cumulative energy", "apparent")
}

#Mapping between the names on the dataset and the indexes used in the nilmtk
appliance_meter_mapping = {
    "688AB5006CFE" : {
        "mains" : 1,
        "fridge" : 2
    },
    "688AB50004EF" : {
        "mains" : 1,
        "fridge" : 2,
        "heat pump": 3
    },
    "006056131261" : {
        "mains" : 1,
        "dish washer" : 2,
        "washing machine" : 3
    },
    "F88A3C900128" : {
        "mains" : 1,
        "fridge" : 2
    }
}

house_mapping = {
    "688AB5006CFE" : 1,
    "688AB50004EF" : 2,
    "006056131261" : 3,
    "F88A3C900128" : 4
}

withus_path = "../../../../datasets/withus_classification/"
metada_path = "./metadata"
output_filename = "../../../../datasets/withus_h5/withus.h5"

convert_withus(withus_path, output_filename)