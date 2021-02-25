
from nilmtk.utils import get_datastore
from nilmtk.utils import get_module_directory, check_directory_exists
from nilmtk.datastore import Key
from nilmtk.measurement import LEVEL_NAMES
from os.path import join, isdir, isfile
from os import listdir
import re
from sys import stdout
from nilm_metadata import save_yaml_to_datastore
import pandas as pd
import numpy as np

def convert_aveiro(aveiro_path, output_filename):
    """
    Parameters
    ----------
    aveiro_path : str
        The root path of the avEiro low_freq dataset.
    output_filename : str
        The destination filename (including path and suffix).
    """

    # Open DataStore
    store = get_datastore(output_filename, "HDF", mode='w')

    # Convert raw data to DataStore
    check_directory_exists(aveiro_path)

    houses = _find_all_houses(aveiro_path)

    for house_id in houses:
        print("Loading house ", house_id)
        stdout.flush()
        for appliance, meter in appliance_meter_mapping.items():
            print("Loading ", appliance)
            stdout.flush()

            key = Key(building=house_id, meter=meter)

            if appliance == "mains":
                dfs = []
                for measure in column_mapping.keys():
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
                
                total = pd.concat(dfs, axis=1)

                for c in total.columns.values:
                    for i in range(0, len(total[c])):
                        if np.isnan(total[c][i]):
                            for j in range(i+1, len(total[c])):
                                if not np.isnan(total[c][j]):
                                    total[c][i] = (total[c][i-1] + total[c][j])/2
                                    break
                        if np.isnan(total[c][i]):
                            for j in range(i, len(total[c])):
                                total[c][j] = total[c][j-1]
                            break
                            
                total = total.tz_localize('UTC').tz_convert('Europe/London')
                total.columns = pd.MultiIndex.from_tuples([column_mapping[x] for x in total.columns])
                total.columns.set_names(LEVEL_NAMES, inplace=True)

                store.put(str(key), total)
            else:
                csv_filename = aveiro_path + "house_" + str(house_id) + "/" + str(appliance) + "/power.csv"
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
                df = df.tz_localize('UTC').tz_convert('Europe/London')
                df.columns = pd.MultiIndex.from_tuples([column_mapping["power"]])
                df.columns.set_names(LEVEL_NAMES, inplace=True)
                store.put(str(key), df)
        print()

    # Add metadata
    save_yaml_to_datastore(metada_path, store)

    store.close()
    print("Done converting avEiro to HDF5!")
    
def _find_all_houses(input_path):
    """
    Returns
    -------
    list of integers (house instances)
    """
    dir_names = [p for p in listdir(input_path) if isdir(join(input_path, p))]
    return _matching_ints(dir_names, r'^house_(\d)$')

def _matching_ints(strings, regex):
    """Uses regular expression to select and then extract an integer from
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

column_mapping = {
    "power" : ("power", "apparent"),
    "vrms" : ("voltage", "")
}

appliance_meter_mapping = {
    "mains" : 1,
    "heatpump" : 2,
    "carcharger" : 3
}

aveiro_path = "../../../datasets/avEiro/"
metada_path = aveiro_path + "metadata"
output_filename = "../../../datasets/avEiro_h5/avEiro.h5"
convert_aveiro(aveiro_path, output_filename)