
import os
import pandas as pd

from nilmtk.measurement import LEVEL_NAMES
from nilmtk import DataSet
import numpy as np


def create_path(path):

    if not os.path.isdir(path):
        os.makedirs(path)


def load_data(dataset_folder, appliance, houses, timestep, mains_columns, appliance_columns):
    #Array that contains the aggragate readings in dataframes
    aggregated_readings = []


    #Array that contains the readings from the appliances
    app_readings = []

    column_mapping = {
        "power" : ("power", "apparent"),
        "voltage" : ("voltage", "")
    }

    #Goes through all the houses
    for house in houses:
        #Loads the mains files
        mains = pd.read_csv(dataset_folder+house+"/mains.csv", sep=',', header=[0,1], index_col=0)
        mains.columns = pd.MultiIndex.from_tuples(mains.columns)
        mains.index = pd.to_datetime(mains.index)
        mains.columns = pd.MultiIndex.from_tuples([column_mapping[x[0]] for x in mains.columns.values])

        #appends that dataframe to an array
        for c in mains.columns:
            if c not in mains_columns:
                mains = mains.drop(c, axis=1)
        

        #Loads the appliance to be used from that house using a similar logic.
        app = pd.read_csv(dataset_folder+house+ "/"+ appliance + ".csv", sep=',', header=[0,1], index_col=0)
        app.index = pd.to_datetime(app.index)
        app.columns = pd.MultiIndex.from_tuples(app.columns)

        #Stores the dataframe in a dictionary
        for c in app.columns:
            if c not in appliance_columns:
                app = app.drop(c, axis=1)

        for timeperiod in houses[house]:
            if timeperiod[0] is None:
                #Intersects the intervals between the mains data and the appliance data.
                if mains.index[0] < app.index[0]:
                    beginning = app.index[0]
                else: 
                    beginning = mains.index[0]
            else:
                if mains.index[0] > timeperiod[0] or app.index[0] > timeperiod[0]:
                    raise Exception("Beginning Time before the beginning of the house dataset, use None instead.")
                elif mains.index[-1] < timeperiod[0] or app.index[-1] < timeperiod[0]:
                    raise Exception("Beginning Time after the end of the house dataset, use None instead.")
                else:
                    beginning = timeperiod[0]

            if timeperiod[1] is None:
                if mains.index[-1] < app.index[-1]:
                    end = mains.index[-1]
                else:
                    end = app.index[-1]
            else:
                if mains.index[-1] < timeperiod[1] or app.index[-1] < timeperiod[1]:
                    raise Exception("End Time after the end of the house dataset, use None instead.")
                elif mains.index[0] > timeperiod[1] or app.index[0] > timeperiod[1]:
                    raise Exception("End Time before the beginning of the house dataset, use None instead.")
                else:
                    end = timeperiod[1]
            if beginning > end:
                    raise Exception("Beginning time is after the end time. Invalid time interval.")

            beginning_index_x = mains.index.get_loc(beginning, method="nearest")
            end_index_x = mains.index.get_loc(end, method="nearest")

            beginning_index_y = app.index.get_loc(beginning, method="nearest")
            end_index_y = app.index.get_loc(end, method="nearest")

            #Updates the dfs to only use the intersection.
            aggregated_readings.append(mains[beginning_index_x: end_index_x])
            app_readings.append(app[beginning_index_y: end_index_y])

    return (aggregated_readings, app_readings)

def load_data_h5(dataset_file, appliance, houses, timestep, mains_columns, appliance_columns, use_activations=False, appliance_params=None):
    #Array that contains the aggragate readings in dataframes
    aggregated_readings = []


    #Array that contains the readings from the appliances
    app_readings = []

    dataset = DataSet(dataset_file)  

    for house in houses:

        for timeperiod in houses[house]:
            
            if not use_activations:

                dataset.set_window(start=timeperiod[0],end=timeperiod[1])

                mains = next(dataset.buildings[house].elec.mains().load(sample_period=timestep))

                for c in mains.columns:
                    if c not in mains_columns:
                        del mains[c]

                app_df = next(dataset.buildings[house].elec[appliance].load(sample_period=timestep))

                for c in app_df.columns:
                    if c not in appliance_columns:
                        del app_df[c]
                
                aggregated_readings.append(mains)
                app_readings.append(app_df)
            else:
                dataset.set_window(start=timeperiod[0],end=timeperiod[1])

                activations = dataset.buildings[house].elec[appliance].activation_series(
                    appliance_params["min_off_time"], 
                    appliance_params["min_on_time"], 
                    appliance_params["number_of_activation_padding"], 
                    appliance_params["min_on_power"])
                
                for period in activations:
                    timestamp = period.index[0]
                    tz_offset = (timestamp.replace(tzinfo=None) - 
                                timestamp.tz_convert('UTC').replace(tzinfo=None))
                    tz_offset_bg = np.timedelta64(tz_offset)

                    timestamp = period.index[-1]
                    tz_offset = (timestamp.replace(tzinfo=None) - 
                                timestamp.tz_convert('UTC').replace(tzinfo=None))

                    tz_offset_end = np.timedelta64(tz_offset)

                    if tz_offset_bg > tz_offset_end:
                        beg = period.index[0].tz_localize(None) - tz_offset_bg
                        end = period.index[-1].tz_localize(None)
                    elif tz_offset_bg < tz_offset_end: 
                        beg = period.index[0].tz_localize(None)
                        end = period.index[-1].tz_localize(None) - tz_offset_end 
                    else:
                        beg = period.index[0].tz_localize(None)
                        end = period.index[-1].tz_localize(None)

                    dataset.set_window(
                            start=beg,
                            end=end
                            )

                    mains_df = next(dataset.buildings[house].elec.mains().load(sample_period=timestep))

                    for c in mains_df.columns:
                            if c not in mains_columns:
                                del mains_df[c]
                    
                    if not mains_df.empty:
                        mains_df = mains_df[[list(mains_df.columns)[0]]]

                        appliance_df = next(dataset.buildings[house].elec[appliance].load(sample_period=timestep))
                        
                        for c in appliance_df.columns:
                            if c not in appliance_columns:
                                del appliance_df[c]
                        
                        appliance_df = appliance_df[[list(appliance_df.columns)[0]]]

                        mains_df, appliance_dfs = dropna(mains_df, [appliance_df])

                        aggregated_readings.append(mains_df)
                        app_readings.append(appliance_dfs[0])      

    return (aggregated_readings, app_readings)


def dropna(mains_df, appliance_dfs):
    """
    Drops the missing values in the Mains reading and appliance readings and returns consistent data by copmuting the intersection
    """
    # The below steps are for making sure that data is consistent by doing intersection across appliances
    mains_df = mains_df.dropna()
    ix = mains_df.index
    mains_df = mains_df.loc[ix]
    for i in range(len(appliance_dfs)):
        appliance_dfs[i] = appliance_dfs[i].dropna()

    for app_df in appliance_dfs:
        ix = ix.intersection(app_df.index)
    mains_df = mains_df.loc[ix]
    new_appliances_list = []
    for app_df in appliance_dfs:
        new_appliances_list.append(app_df.loc[ix])
    return mains_df,new_appliances_list