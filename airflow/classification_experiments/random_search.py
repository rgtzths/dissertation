import datetime
import warnings
warnings.filterwarnings("ignore")

from scipy.stats import randint, loguniform


import sys
sys.path.insert(1, "/home/rteixeira/thesis/feature_extractors")
sys.path.insert(1, "/home/rteixeira/thesis/classification_models")

import dataset_loader
from gru_dwt import GRU_DWT

import sys
import logging

def run_experiment(dwt_timewindow, dwt_overlap, examples_overlap, examples_timewindow, wavelet):

    #Experiment Definition
    experiment = {
        "heatpump" : {
            "methods" : {
                "GRU_DWT" :GRU_DWT( {
                    "verbose" : 2,
                    "appliances" : {
                        "heatpump" : {
                            "timestep" : 2,
                            "dwt_timewindow" : dwt_timewindow,
                            "dwt_overlap" : dwt_overlap,
                            "examples_overlap" : examples_overlap,
                            "examples_timewindow" : examples_timewindow,
                            "epochs" : 1,
                            "batch_size" : 2000,
                            "wavelet": 'bior2.2'
                        }
                    },
                    "training_results_path" : "/home/rteixeira/outputs/models/gru_dwt/",
                    "predicted_column": ("power", "apparent"), 
                    "randomsearch": True,
                    "randomsearch_params": {
                        "file_path" : "/home/rteixeira/outputs/random_search_results/randomsearch_results.csv",
                        "n_nodes" : (0.5, 2),
                        "n_iter" : 50,
                        "n_jobs" : 5,
                        "cv" : 5,
                        "model": {
                            "epochs" : randint(1,2),
                            "batch_size" : randint(500,1000),
                        }
                    }
                }),
            },
            "model_path" : "./models/",
            "train" : {
                "ukdale" : {
                    "location" : "/home/rteixeira/datasets/avEiro_classification/",
                    "houses" : {
                        "house_1" : {
                            "beginning" : datetime.datetime(2020, 10, 1),
                            "end" : datetime.datetime(2020, 10, 2)
                        }
                    }
                },
            },
            "test" : {
                "ukdale" : {
                    "location" : "/home/rteixeira/datasets/avEiro_classification/",
                    "houses" : {
                        "house_1" : {
                            "beginning" : datetime.datetime(2021, 1, 15),
                            "end" : datetime.datetime(2021, 1, 16)
                        }
                    }
                }
            }
        }
    }

    results = {}
    for app in experiment:
        X_train = []
        y_train = []

        X_test = []
        y_test = []

        results[app] = {}

        print("Loading Train Data for %s" % (app))
        for dataset in experiment[app]["train"]:
            x, y = dataset_loader.load_data(
                        experiment[app]["train"][dataset]["location"],
                        app, 
                        experiment[app]["train"][dataset]["houses"]
                    )
            for i in range(0, len(x)):
                X_train.append(x[i])
                y_train.append(y[i])
                
        

        for method in experiment[app]["methods"]:
            print("Training %s" % (method))
            experiment[app]["methods"][method].partial_fit(X_train, [(app, y_train)])

            experiment[app]["methods"][method].save_model(experiment[app]["model_path"] + method.lower())

if __name__ == "__main__":
    run_experiment(8, 4, 150, 300, 'bior2.2')
