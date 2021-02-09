from os.path import join, isdir, isfile
from os import listdir
import re
from sys import stdout
import pandas as pd
import numpy as np


column_mapping = {
    "power" : ("power", "apparent"),
    "vrms" : ("voltage", "")
}

appliance_meter_mapping = {
    "mains" : 1,
    "heatpump" : 2,
    "carcharger" : 3
}



def convert_aveiro(aveiro_path, output_path, timeframe, timestep):
    """
    Parameters
    ----------
    aveiro_path : str
        The root path of the avEiro low_freq dataset.
    output_filename : str
        The destination filename (including path and suffix).
    timeframe : int
        Time covered in each frame in min
    """

    houses = _find_all_houses(aveiro_path)

    one_second = pd.Timedelta(1, unit="s")
    step = pd.Timedelta(timestep, unit="s")

    for house_id in houses:
        print("Loading house ", house_id)
        stdout.flush()

        filenames = []

        for appliance, meter in appliance_meter_mapping.items():
            print("Converting ", appliance)
            stdout.flush()

            dfs = []
            if appliance == "mains":
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
            else:
                measure = "power"
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

            df = pd.concat(dfs, axis=1)
            data = []
            
            columns = list(df.columns.values)

            current_time = dfs[0].index[0]
            current_index = 0
            objective_time = current_time + step

            past_feature_vector = []
            aprox = 0
            arred = 0
            behind = 0

            while current_index < len(df):

                feature_vector = []
                if len(past_feature_vector) != 0:
                    feature_vector = past_feature_vector[ 1:]
                elif len(past_feature_vector) == 0:
                    feature_vector = [0 for i in range(0, int(timeframe*60*len(columns)/timestep -1))]

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
                    objective_time += objective_step
                    past_feature_vector = feature_vector
                    if appliance == "mains":
                        data.append(feature_vector)
                    else:
                        added = False
                        for i in feature_vector:
                            if i > 0:
                                data.append(1)
                                added = True
                                break
                        if not added:
                            data.append(0)
            
            new_df = pd.DataFrame(data)
            new_df.to_csv('{}/house_{}/{}.csv'.format(output_path, house_id, appliance), header=False)

            #while current_index < len(df):
#
            #    feature_vector = []
            #    if len(past_feature_vector) != 0:
            #        feature_vector = past_feature_vector[ 1:]
#
            #    while current_time != objective_time and current_index < len(df):
            #        index_time = df.index[current_index].round("s", ambiguous=False)
            #        if index_time == current_time or index_time - one_second == current_time or index_time + one_second == current_time:
            #            feature_vector.append(column[df.index[current_index]])
            #            current_index += 1
            #            current_time += timestep
            #            aprox += 1
            #        elif current_time > index_time:
            #            next_index = df.index[current_index+1].round("s", ambiguous=False)
            #            if next_index == current_time or next_index - one_second == current_time or next_index + one_second == current_time:
            #                feature_vector.append(column[df.index[current_index+1]])
            #                current_time += timestep
            #            current_index += 2
            #            behind += 1
            #        else:
            #            feature_vector.append( (column[df.index[current_index]] + feature_vector[-1])/2 )
            #            current_time += timestep
            #            arred += 1
            #    if current_index < len(df):
            #        objective_time += timestep
            #        past_feature_vector = feature_vector
            #        stdout.flush()
            #        if chan_id == 1:
            #            data.append(feature_vector)
            #        else:
            #            added = False
            #            for i in feature_vector:
            #                if i > 0:
            #                    data.append(1)
            #                    added = True
            #                    break
            #            if not added:
            #                data.append(0)
            print("")
           
            print("Aprox Values: ", aprox)
            print("Arred values: ", arred)
            print("Behind values:", behind)
            print("")
            #new_df = pd.DataFrame(data)
            #new_df.to_csv('{}/house_{}/channel_{}.csv'.format(output_path, house_id, chan_id), index = False, header=False)
            
        print()

    print("Done converting avEiro to Timeseries!")


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


filespath = "../../../datasets/avEiro_dataset_v2/"
output_path = "../../../datasets/avEiro_timeseries"

timeframe = 10
timestep = 2

convert_aveiro(filespath, output_path, timeframe, timestep)