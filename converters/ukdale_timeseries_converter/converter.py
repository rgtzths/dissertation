from os.path import join, isdir, isfile
from os import listdir
import re
from sys import stdout
import pandas as pd
import numpy as np

house_appliances_mappings = {
    "house1" : {
        "aggregate" : "1",
        "washing_machine": "5",
        "dish_washer": "6",
        "fridge": "12",
        "microwave": "13",
        "boiler": "2"
    },
    "house2" : {
        "aggregate" : "1",
        "washing_machine": "12",
        "dish_washer": "13",
        "kettle": "8",
        "toaster": "16",
        "fridge": "14",
        "microwave": "15",
    },
    "house3" : {
        "aggregate" : "1",
        "kettle": "2",
    },
    "house5" : {
        "aggregate" : "1",
        "oven": "20",
        "dish_washer": "22",
        "kettle": "18",
        "toaster":"15",
        "fridge": "19",
        "microwave": "23",
    },
    "house4" : {}
}



def convert_ukdale(ukdale_path, output_path, timeframe, timestep, overlap):
    """
    Parameters
    ----------
    ukdale_path : str
        The root path of the ukdale low_freq dataset.
    output_filename : str
        The destination filename (including path and suffix).
    timeframe : int
        Time covered in each frame in min
    """

    houses = _find_all_houses(ukdale_path)

    one_second = pd.Timedelta(1, unit="s")
    
    objective_step = pd.Timedelta(timeframe*60, unit="s") - pd.Timedelta(timeframe*60*overlap, unit="s")

    step = pd.Timedelta(timestep, unit="s")

    for house_id in houses:
        print("Loading house ", house_id)
        stdout.flush()

        filenames = []

        for appliance, meter in house_appliances_mappings["house"+str(house_id)].items():
            print("Converting ", appliance)
            stdout.flush()

            csv_filename = ukdale_path + "house_" + str(house_id) + "/channel_" + meter + ".dat"
                    
            measure = "power"

            df = pd.read_csv(csv_filename, sep=" ", header=None)
            df.columns = ["time", measure]
            df.index = pd.to_datetime(df["time"], unit='s')
            
            df = df.drop("time", 1)
            df = df.sort_index()
            
            dups_in_index = df.index.duplicated(keep='first')
            if dups_in_index.any():
                df = df[~dups_in_index]

            data = []
            
            columns = list(df.columns.values)

            current_time = df.index[0]
            current_index = 0
            objective_time = current_time + objective_step*2
            overlap_index = int(timeframe*60/timestep - timeframe*60*overlap*len(columns)/timestep)

            past_feature_vector = []
            aprox = 0
            arred = 0
            behind = 0
            
            while current_index < len(df):

                feature_vector = []
                if len(past_feature_vector) != 0:
                    feature_vector = past_feature_vector[overlap_index:]

                while current_time != objective_time and current_index < len(df):
                    index_time = df.index[current_index].round("s", ambiguous=False)
                    if index_time == current_time or index_time - one_second == current_time or index_time + one_second == current_time:
                        for c in columns:
                            feature_vector.append(df[c][df.index[current_index]])
                        current_index += 1
                        current_time += step
                        aprox += 1
                    elif current_time > index_time:
                        if  current_index < len(df) -1:
                            next_index = df.index[current_index+1].round("s", ambiguous=False)
                            if next_index == current_time or next_index - one_second == current_time:
                                for c in columns:
                                    feature_vector.append(df[c][df.index[current_index+1]])
                                current_time += step
                                behind += 1
                        current_index += 2
                    else:
                        for c in columns:
                            feature_vector.append((df[c][df.index[current_index]] + feature_vector[-len(columns)])/2)
                        current_time += step
                        arred += 1

                if len(feature_vector) == int(timeframe*60*len(columns)/timestep):
                    
                    past_feature_vector = feature_vector
                    if appliance == "mains":
                        data.append([objective_time]+ feature_vector )
                    else:
                        added = False
                        for i in feature_vector:
                            if i > 0:
                                data.append([objective_time, 1])
                                added = True
                                break
                        if not added:
                            data.append([objective_time, 0])

                    objective_time += objective_step
            
            new_df = pd.DataFrame(data)
            new_df = new_df.set_index(0)
            new_df.to_csv('{}/house_{}/{}.csv'.format(output_path, house_id, appliance), header=False)
            
            print("")
            print("Aprox Values: ", aprox)
            print("Arred values: ", arred)
            print("Behind values:", behind)
            print()

        print()

    print("Done converting UKDale to Timeseries!")


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


filespath = "../../../datasets/ukdale/"
output_path = "../../../datasets/ukdale_timeseries"

timeframe = 10
timestep = 6
overlap = 0.5

convert_ukdale(filespath, output_path, timeframe, timestep, overlap)