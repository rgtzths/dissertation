from os import remove, listdir
from os.path import join, isdir, isfile
from sys import stdout
import re

from nilmtk.utils import check_directory_exists
from nilmtk import DataSet
from nilmtk.utils import get_datastore
from nilmtk.datastore import Key
from nilm_metadata import convert_yaml_to_hdf5
from nilmtk.measurement import LEVEL_NAMES

import pandas as pd
import numpy as np

import sys
sys.path.insert(1, "../../")
sys.path.insert(1, "../../../utils")

ONE_SEC_COLUMNS = [('power', 'active'), ('power', 'apparent'), ('voltage', '')]
TZ = 'Europe/London'

from data_clean import clean_data
import utils

def convert_ukdale(ukdale_path, output_filename, format='HDF', drop_duplicates=True):
    """Converts the UK-DALE dataset to NILMTK HDF5 format.

    For more information about the UK-DALE dataset, and to download
    it, please see http://www.doc.ic.ac.uk/~dk3810/data/

    Parameters
    ----------
    ukdale_path : str
        The root path of the UK-DALE dataset.  It is assumed that the YAML
        metadata is in 'ukdale_path/metadata'.
    output_filename : str
        The destination filename (including path and suffix).
    format : str
        format of output. Either 'HDF' or 'CSV'. Defaults to 'HDF'
    drop_duplicates : bool
        Remove entries with duplicated timestamp (keeps the first value)
        Defaults to True.
    """
    base_path = ""
    for directory in  output_filename.split("/")[0:-1]:
        base_path += directory + "/"
    utils.create_path( base_path)

    ac_type_map = _get_ac_type_map(ukdale_path)

    def _ukdale_measurement_mapping_func(house_id, chan_id):
        ac_type = ac_type_map[(house_id, chan_id)][0]
        return [('power', ac_type)]

    # Open DataStore
    store = get_datastore(output_filename, format, mode='w')

    # Convert 6-second data
    _convert(ukdale_path, store, _ukdale_measurement_mapping_func, TZ,
             sort_index=False, drop_duplicates=drop_duplicates)
    store.close()

    # Add metadata
    if format == 'HDF':
        convert_yaml_to_hdf5(join(ukdale_path, 'metadata'), output_filename)

    store.close()
    print("Done converting UK-DALE to HDF5!")


def _get_ac_type_map(ukdale_path):
    """First we need to convert the YAML metadata to HDF5
    so we can load the metadata into NILMTK to allow
    us to use NILMTK to find the ac_type for each channel.
    
    Parameters
    ----------
    ukdale_path : str

    Returns
    -------
    ac_type_map : dict.  
        Keys are pairs of ints: (<house_instance>, <meter_instance>)
        Values are list of available power ac type for that meter.
    """

    hdf5_just_metadata = join(ukdale_path, 'metadata', 'ukdale_metadata.h5')
    convert_yaml_to_hdf5(join(ukdale_path, 'metadata'), hdf5_just_metadata)
    ukdale_dataset = DataSet(hdf5_just_metadata)
    ac_type_map = {}
    for building_i, building in ukdale_dataset.buildings.items():
        elec = building.elec
        for meter in elec.meters + elec.disabled_meters:
            key = (building_i, meter.instance())
            ac_type_map[key] = meter.available_ac_types('power')

    ukdale_dataset.store.close()
    remove(hdf5_just_metadata)
    return ac_type_map

def _load_csv(filename, columns, tz, drop_duplicates=False, sort_index=False):
    """
    Parameters
    ----------
    filename : str
    columns : list of tuples (for hierarchical column index)
    tz : str 
        e.g. 'US/Eastern'
    sort_index : bool
        Defaults to True
    drop_duplicates : bool
        Remove entries with duplicated timestamp (keeps the first value)
        Defaults to False for backwards compatibility.

    Returns
    -------
    pandas.DataFrame
    """
    # Load data
    df = pd.read_csv(filename, sep=' ', names=columns,
                     dtype={m:np.float32 for m in columns})
    
    # Modify the column labels to reflect the power measurements recorded.
    df.columns.set_names(LEVEL_NAMES, inplace=True)

    # Convert the integer index column to timezone-aware datetime 
    df.index = pd.to_datetime(df.index.values, unit='s', utc=True)
    df = df.tz_convert(tz)

    if sort_index:
        df = df.sort_index() # raw REDD data isn't always sorted
        
    if drop_duplicates:
        dups_in_index = df.index.duplicated(keep='first')
        if dups_in_index.any():
            df = df[~dups_in_index]

    return df

def _convert(input_path, store, measurement_mapping_func, tz, sort_index=True, drop_duplicates=False):
    """
    Parameters
    ----------
    input_path : str
        The root path of the REDD low_freq dataset.
    store : DataStore
        The NILMTK DataStore object.
    measurement_mapping_func : function
        Must take these parameters:
            - house_id
            - chan_id
        Function should return a list of tuples e.g. [('power', 'active')]
    tz : str 
        Timezone e.g. 'US/Eastern'
    sort_index : bool
        Defaults to True
    drop_duplicates : bool
        Remove entries with duplicated timestamp (keeps the first value)
        Defaults to False for backwards compatibility.
    """

    check_directory_exists(input_path)

    # Iterate though all houses and channels
    houses = _find_all_houses(input_path)
    for house_id in houses:
        print("Loading house", house_id, end="... ")
        stdout.flush()
        chans = _find_all_chans(input_path, house_id)
        for chan_id in chans:
            print(chan_id, end=" ")
            stdout.flush()
            key = Key(building=house_id, meter=chan_id)
            measurements = measurement_mapping_func(house_id, chan_id)
            csv_filename = _get_csv_filename(input_path, key)
            df = _load_csv(csv_filename, measurements, tz, 
                sort_index=sort_index, 
                drop_duplicates=drop_duplicates
            )
            
            df = clean_data(df, 6, "previous")

            store.put(str(key), df)
        print()

def _get_csv_filename(input_path, key_obj):
    """
    Parameters
    ----------
    input_path : (str) the root path of the REDD low_freq dataset
    key_obj : (nilmtk.Key) the house and channel to load

    Returns
    ------- 
    filename : str
    """
    assert isinstance(input_path, str)
    assert isinstance(key_obj, Key)

    # Get path
    house_path = 'house_{:d}'.format(key_obj.building)
    path = join(input_path, house_path)
    assert isdir(path)

    # Get filename
    filename = 'channel_{:d}.dat'.format(key_obj.meter)
    filename = join(path, filename)
    assert isfile(filename)

    return filename

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
    return _matching_ints(filenames, r'^channel_(\d\d?).dat$')


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

if __name__ == "__main__":
    convert_ukdale("../../../../datasets/ukdale/", "../../../../datasets/ukdale_h5/ukdale.h5" , format='HDF', drop_duplicates=True)