
import pandas as pd
import numpy as np
import datetime
from matplotlib import pyplot as plt

from nilmtk.measurement import LEVEL_NAMES


experiment = {
    "fridge" : {
        "location" : "../../datasets/ukdale_classification/",
        "houses" : {
            "house_1" : [
                (datetime.datetime(2014, 2, 20), datetime.datetime(2014, 2, 28)),
                (datetime.datetime(2016, 9, 10), datetime.datetime(2016, 9, 17))
            ],
            "house_2" : [
                (datetime.datetime(2013, 5, 23), datetime.datetime(2013, 5, 30)),
                (datetime.datetime(2013, 7, 10), datetime.datetime(2013, 7, 17))
            ],
            "house_5" : [
                (datetime.datetime(2014, 8, 1),datetime.datetime(2014, 8, 8)),
                (datetime.datetime(2014, 9, 8),datetime.datetime(2014, 9, 15))
            ]
        }
        
    },
    "microwave" : {
        "location" : "../../datasets/ukdale_classification/",
        "houses" : {
            "house_1" : [
                (datetime.datetime(2014, 2, 20), datetime.datetime(2014, 2, 28)),
                (datetime.datetime(2016, 9, 10), datetime.datetime(2016, 9, 17))
            ],
            "house_2" : [
                (datetime.datetime(2013, 5, 23), datetime.datetime(2013, 5, 30)),
                (datetime.datetime(2013, 7, 10), datetime.datetime(2013, 7, 17))
            ],
            "house_5" : [
                (datetime.datetime(2014, 8, 1),datetime.datetime(2014, 8, 8)),
                (datetime.datetime(2014, 9, 8),datetime.datetime(2014, 9, 15))
            ]
        }  
    },
    "dish_washer" : {
        "location" : "../../datasets/ukdale_classification/",
        "houses" : {
            "house_1" : [
                (datetime.datetime(2014, 2, 20), datetime.datetime(2014, 2, 28)),
                (datetime.datetime(2016, 9, 10), datetime.datetime(2016, 9, 17))
            ],
            "house_2" : [
                (datetime.datetime(2013, 9, 23), datetime.datetime(2013, 9, 30)),
                (datetime.datetime(2013, 6, 10), datetime.datetime(2013, 6, 17))
            ],
            "house_5" : [
                (datetime.datetime(2014, 8, 1),datetime.datetime(2014, 8, 8)),
                (datetime.datetime(2014, 9, 8),datetime.datetime(2014, 9, 15))
            ]
        }
        
    }
}

column_mapping = {
    "power" : ("power", "apparent"),
    "voltage" : ("voltage", "")
}

for app in experiment:
    for house in experiment[app]["houses"]:

        mains = pd.read_csv(experiment[app]["location"]+house+"/mains.csv", sep=',', header=[0,1], index_col=0)
        mains.columns = pd.MultiIndex.from_tuples(mains.columns)
        mains.index = pd.to_datetime(mains.index)
        mains.columns = pd.MultiIndex.from_tuples([column_mapping[x[0]] for x in mains.columns.values])

        app_df = pd.read_csv(experiment[app]["location"]+house+ "/"+ app + ".csv", sep=',', header=[0,1], index_col=0)
        app_df.index = pd.to_datetime(app_df.index)
        app_df.columns = pd.MultiIndex.from_tuples(app_df.columns)
        
        for timeperiod in experiment[app]["houses"][house]:

            mains_beginning_index = mains.index.get_loc(timeperiod[0], method="nearest")

            mains_end_index = mains.index.get_loc(timeperiod[1], method="nearest")

            app_beginning_index = app_df.index.get_loc(timeperiod[0], method="nearest")

            app_end_index = app_df.index.get_loc(timeperiod[1], method="nearest")

            plt.plot(mains.index[mains_beginning_index:mains_end_index], mains["power"]["apparent"][mains_beginning_index:mains_end_index], label="Mains Energy")
            plt.plot(app_df.index[app_beginning_index:app_end_index], app_df["power"]["apparent"][app_beginning_index:app_end_index], label=app + " Energy")
            plt.xlabel("Time")
            plt.ylabel("Power")
            plt.title("Mains and " + app + " energy consumption from " + house)
            plt.legend()

            plt.show()