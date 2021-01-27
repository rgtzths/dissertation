from os.path import join, isdir, isfile
from os import listdir
import re
from sys import stdout
import pandas as pd
import numpy as np


def convert_aveiro(aveiro_path, output_path, timeframe, overlap):
    """
    Parameters
    ----------
    aveiro_path : str
        The root path of the avEiro low_freq dataset.
    output_filename : str
        The destination filename (including path and suffix).
    timeframe : int
        Time covered in each frame in min
    overlap : float
        Overlaping between timeframes (between 0 and 1)
    """
    columns = [('power', "apparent")]

    houses = _find_all_houses(aveiro_path)
    one_second = pd.Timedelta(1, unit="s")
    two_seconds = pd.Timedelta(2, unit="s")
    overlap_index =  timeframe*30 - int(timeframe*30*overlap) -1
    objective_step = pd.Timedelta(timeframe*60, unit="s") - pd.Timedelta(timeframe*60*overlap, unit="s")

    for house_id in houses:
        print("Loading house", house_id, end="...\n")
        stdout.flush()
        chans = _find_all_chans(aveiro_path, house_id)
        filenames = []

        
        
        for chan_id in chans:
            print("Converting Channel", chan_id, end=" \n")
            stdout.flush()

            csv_filename = _get_csv_filename(aveiro_path, house_id, chan_id)
            df = pd.read_csv(csv_filename, sep=',', names=columns, 
                    dtype={m:np.float32 for m in columns})

            df.index = pd.to_datetime(df.index.values, unit='ns', utc=True)
            df = df.tz_convert("Europe/London")
            
            df = df.sort_index()

            column = df["power"]["apparent"]
            data = []
            
            current_time = df.index[0].round("s", ambiguous=False)
            current_index = 0
            objective_time = current_time + pd.Timedelta(timeframe*60, unit="s")
            past_feature_vector = []
            aprox = 0
            arred = 0
            behind = 0
            while current_index < len(df):
                feature_vector = []
                if len(past_feature_vector) != 0:
                    feature_vector = past_feature_vector[ overlap_index : -1]

                while current_time != objective_time and current_index < len(df):
                    index_time = df.index[current_index].round("s", ambiguous=False)
                    if index_time == current_time or index_time - one_second == current_time or index_time + one_second == current_time:
                        feature_vector.append(column[df.index[current_index]])
                        current_index += 1
                        current_time += two_seconds
                        aprox += 1
                    elif current_time > index_time:
                        next_index = df.index[current_index+1].round("s", ambiguous=False)
                        if next_index == current_time or next_index - one_second == current_time or next_index + one_second == current_time:
                            feature_vector.append(column[df.index[current_index+1]])
                            current_time += two_seconds
                        current_index += 2
                        behind += 1
                    else:
                        feature_vector.append( (column[df.index[current_index]] + feature_vector[-1])/2 )
                        current_time += two_seconds
                        arred += 1
                if current_index < len(df):
                    objective_time += objective_step
                    past_feature_vector = feature_vector
                    stdout.flush()
                    if chan_id == 1:
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
            print("")
           
            print("Aprox Values: ", aprox)
            print("Arred values: ", arred)
            print("Behind values:", behind)
            print("")
            new_df = pd.DataFrame(data)
            new_df.to_csv('{}/house_{}/channel_{}.csv'.format(output_path, house_id, chan_id), index = False, header=False)
            
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


def _find_all_chans(input_path, house_id):
    """
    Returns
    -------
    list of integers (channels)
    """
    house_path = join(input_path, 'house_{:d}'.format(house_id))
    filenames = [p for p in listdir(house_path) if isfile(join(house_path, p))]
    return _matching_ints(filenames, r'^channel_(\d\d?).csv$')


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

def _get_csv_filename(input_path, house_id, chan_id):
    """
    Parameters
    ----------
    input_path : (str) the root path of the aveiro low_freq dataset
    house_id : (int) house number
    chan_id : (int) channel number

    Returns
    ------- 
    filename : str
    """
    assert isinstance(input_path, str)
    assert isinstance(house_id, int)
    assert isinstance(chan_id, int)

    # Get path
    house_path = 'house_{:d}'.format(house_id)
    path = join(input_path, house_path)
    assert isdir(path)

    # Get filename
    filename = 'channel_{:d}.csv'.format(chan_id)
    filename = join(path, filename)
    assert isfile(filename)

    return filename


filespath = "../avEiro_dataset/"
output_path = "../avEiro_timeseries"

timeframe = 1
overlap = 0.5

convert_aveiro(filespath, output_path, timeframe, overlap)