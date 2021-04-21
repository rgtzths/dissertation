import datetime
import warnings
warnings.filterwarnings("ignore")

from scipy.stats import randint, loguniform

from lstm import LSTM_RNN
from gru import GRU_RNN
from gradient import GradientBoosting
from cnn import CNN
from svm import SVM
from gru_dwt import GRU_DWT

import sys
sys.path.insert(1, "../feature_extractors")
import dataset_loader

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
                            "epochs" : 200,
                            "batch_size" : 2000,
                            "wavelet": 'bior2.2'
                        }
                    },
                    "training_results_path" : "./models/gru_dwt/",
                    "predicted_column": ("power", "apparent"), 
                    "randomsearch": True,
                    "randomsearch_params": {
                        "file_path" : "./random_search_results/randomsearch_results.csv",
                        "n_nodes" : (0.5, 2),
                        "n_iter" : 50,
                        "n_jobs" : -1,
                        "cv" : 5,
                        "model": {
                            "epochs" : randint(1,200),
                            "batch_size" : randint(500,1000),
                        }
                    }
                }),
            },
            "model_path" : "./models/",
            "train" : {
                "ukdale" : {
                    "location" : "../../datasets/avEiro_classification/",
                    "houses" : {
                        "house_1" : {
                            "beginning" : datetime.datetime(2020, 10, 1),
                            "end" : datetime.datetime(2020, 12, 1)
                        }
                    }
                },
            },
            "test" : {
                "ukdale" : {
                    "location" : "../../datasets/avEiro_classification/",
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

        print("Loading Test Data for %s" % (app))
        for dataset in experiment[app]["test"]:
            x, y = dataset_loader.load_data(
                        experiment[app]["test"][dataset]["location"],
                        app, 
                        experiment[app]["test"][dataset]["houses"]
            )
            for i in range(0, len(x)):
                X_test.append(x[i])
                y_test.append(y[i])
                
        for method in experiment[app]["methods"]:
            print("Testing %s" % (method))
            res = experiment[app]["methods"][method].disaggregate_chunk(X_test, [(app, y_test)])

            results[app][method] = res

    for app in experiment:
        print("Results Obtained for %s" % (app))
        for method in results[app]:
            print("%10s" % (method), end="")
            print("%10.2f" % (results[app][method][app]), end="")
            print()
        print()

if __name__ == "__main__":
    run_experiment(8, 0, 150, 300, 'bior2.2')
