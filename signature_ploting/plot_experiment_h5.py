
import pandas as pd
import numpy as np
import datetime
from matplotlib import pyplot as plt

from nilmtk.measurement import LEVEL_NAMES
from nilmtk import DataSet

experiment = {
    "fridge" : {
        "location" : '../../datasets/ukdale/ukdale.h5',
        "houses" : {
            1 : [
                (datetime.datetime(2014, 2, 20), datetime.datetime(2014, 2, 28)),
                (datetime.datetime(2016, 9, 10), datetime.datetime(2016, 9, 17))
            ],
            2 : [
                (datetime.datetime(2013, 5, 23), datetime.datetime(2013, 5, 30)),
                (datetime.datetime(2013, 7, 10), datetime.datetime(2013, 7, 17))
            ],
            5 : [
                (datetime.datetime(2014, 8, 1),datetime.datetime(2014, 8, 8)),
                (datetime.datetime(2014, 9, 8),datetime.datetime(2014, 9, 15))
            ]
        }
        
    },
    "microwave" : {
        "location" : '../../datasets/ukdale/ukdale.h5',
        "houses" : {
            1 : [
                (datetime.datetime(2014, 2, 20), datetime.datetime(2014, 2, 28)),
                (datetime.datetime(2016, 9, 10), datetime.datetime(2016, 9, 17))
            ],
            2 : [
                (datetime.datetime(2013, 5, 23), datetime.datetime(2013, 5, 30)),
                (datetime.datetime(2013, 7, 10), datetime.datetime(2013, 7, 17))
            ],
            5 : [
                (datetime.datetime(2014, 8, 1),datetime.datetime(2014, 8, 8)),
                (datetime.datetime(2014, 9, 8),datetime.datetime(2014, 9, 15))
            ]
        }  
    },
    "dish washer" : {
        "location" : '../../datasets/ukdale/ukdale.h5',
        "houses" : {
            1 : [
                (datetime.datetime(2014, 2, 20), datetime.datetime(2014, 2, 28)),
                (datetime.datetime(2016, 9, 10), datetime.datetime(2016, 9, 17))
            ],
            2 : [
                (datetime.datetime(2013, 9, 23), datetime.datetime(2013, 9, 30)),
                (datetime.datetime(2013, 6, 10), datetime.datetime(2013, 6, 17))
            ],
            5 : [
                (datetime.datetime(2014, 8, 1),datetime.datetime(2014, 8, 8)),
                (datetime.datetime(2014, 9, 8),datetime.datetime(2014, 9, 15))
            ]
        }
        
    }
}

#experiment = {
#    "fridge" : {
#        "location" : '../../datasets/ampds2/AMPds2.h5',
#        "houses" : {
#            1 : [
#                ("2012-4-4", "2012-4-11"),
#                ("2012-4-20", "2012-4-29")
#            ],
#        }
#        
#    },
#    "heat pump" : {
#        "location" : '../../datasets/ampds2/AMPds2.h5',
#        "houses" : {
#            1 : [
#                ("2012-4-4", "2012-4-11"),
#                ("2012-4-20", "2012-4-29")
#            ],
#        }  
#    },
#    "electric oven" : {
#        "location" : '../../datasets/ampds2/AMPds2.h5',
#        "houses" : {
#            1 : [
#                ("2012-4-4", "2012-4-11"),
#                ("2012-4-17", "2012-4-24")
#            ],
#        }
#        
#    }
#}

#experiment = {
#    "fridge" : {
#        "location" : '../../datasets/iAWE/iawe.h5',
#        "houses" : {
#            1 : [
#                ("2013-6-15", "2013-6-22"),
#                ("2013-7-1", "2013-7-8")
#            ],
#        }
#        
#    },
#    "air conditioner" : {
#        "location" : '../../datasets/iAWE/iawe.h5',
#        "houses" : {
#            1 : [
#                ("2013-6-22", "2013-6-29"),
#                ("2013-7-1", "2013-7-8")
#            ],
#        }  
#    },
#    "washing machine" : {
#        "location" : '../../datasets/iAWE/iawe.h5',
#        "houses" : {
#            1 : [
#                ("2013-6-15", "2013-6-22"),
#                ("2013-7-1", "2013-7-8")
#            ],
#        }
#        
#    }
#}

for app in experiment:
    dataset = DataSet(experiment[app]["location"]) 

    for house in experiment[app]["houses"]:

        for timeperiod in experiment[app]["houses"][house]:
            
            dataset.set_window(start=timeperiod[0],end=timeperiod[1])

            mains = next(dataset.buildings[house].elec.mains().load())

            app_df = next(dataset.buildings[house].elec[app].load())


            plt.plot(mains.index, mains["power"]["apparent"], label="Mains Energy")
            plt.plot(app_df.index, app_df["power"]["active"], label=app + " Energy")
            plt.xlabel("Time")
            plt.ylabel("Power")
            plt.title("Mains and " + app + " energy consumption from house" + str(house))
            plt.legend()

            plt.show()