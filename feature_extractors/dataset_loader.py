import numpy as np
import pandas as pd
from nilmtk.measurement import LEVEL_NAMES

def load_data(dataset_folder, appliance, houses, timestep):
    #Array that contains the aggragate readings in dataframes
    aggregated_readings = []

    #Dictionary that contains the readings from the appliances
    # With format [ df-house1, df-house2... ]
    app_readings = []

    column_mapping = {
        "power" : ("power", "apparent"),
        "vrms" : ("voltage", "")
    }

    #Goes through all the houses
    for house in houses:
        #Loads the mains files
        df = pd.read_csv(dataset_folder+house+"/mains.csv", sep=',', header=[0,1], index_col=0)
        df.columns = pd.MultiIndex.from_tuples(df.columns)
        df.index = pd.to_datetime(df.index)
        df.columns = pd.MultiIndex.from_tuples([column_mapping["power"], column_mapping["vrms"]])
        df.columns.set_names(LEVEL_NAMES, inplace=True)
        #appends that dataframe to an array
        aggregated_readings.append(df)

        #Loads the appliance to be used from that house using a similar logic.
        df = pd.read_csv(dataset_folder+house+ "/"+ appliance + ".csv", sep=',', header=[0,1], index_col=0)
        df.index = pd.to_datetime(df.index)
        df.columns = pd.MultiIndex.from_tuples([column_mapping["power"]])
        df.columns.set_names(LEVEL_NAMES, inplace=True)
        #Stores the dataframe in a dictionary
        app_readings.append(df)
    
    #Goes through all the houses.
    for i, house in enumerate(houses.keys()):
        if houses[house]["beginning"] is None:
            #Intersects the intervals between the mains data and the appliance data.
            if aggregated_readings[i].index[0] < app_readings[i].index[0]:
                beginning = app_readings[i].index[0]
            else: 
                beginning = aggregated_readings[i].index[0]
        else:
            if aggregated_readings[i].index[0] > houses[house]["beginning"] or app_readings[i].index[0] > houses[house]["beginning"]:
                raise Exception("Beginning Time before the beginning of the house dataset, use None instead.")
            elif aggregated_readings[i].index[-1] < houses[house]["beginning"] or app_readings[i].index[-1] < houses[house]["beginning"]:
                raise Exception("Beginning Time after the end of the house dataset, use None instead.")
            else:
                beginning = houses[house]["beginning"]

        if houses[house]["end"] is None:
            if aggregated_readings[i].index[-1] < app_readings[i].index[-1]:
                end = aggregated_readings[i].index[-1]
            else:
                end = app_readings[i].index[-1]
        else:
            if aggregated_readings[i].index[-1] < houses[house]["end"] or app_readings[i].index[-1] < houses[house]["end"]:
                raise Exception("End Time after the end of the house dataset, use None instead.")
            elif aggregated_readings[i].index[0] > houses[house]["end"] or app_readings[i].index[0] > houses[house]["end"]:
                raise Exception("End Time before the beginning of the house dataset, use None instead.")
            else:
                end = houses[house]["end"]
        if beginning > end:
                raise Exception("Beginning time is after the end time. Invalid time interval.")

        beginning_index_x = aggregated_readings[i].index.get_loc(beginning, method="nearest")
        end_index_x = aggregated_readings[i].index.get_loc(end, method="nearest")

        beginning_index_y = app_readings[i].index.get_loc(beginning, method="nearest")
        end_index_y = app_readings[i].index.get_loc(end, method="nearest")

        #Updates the dfs to only use the intersection.
        aggregated_readings[i] = aggregated_readings[i][beginning_index_x: end_index_x]
        app_readings[i] = app_readings[i][beginning_index_y: end_index_y]

    return (aggregated_readings, app_readings)

