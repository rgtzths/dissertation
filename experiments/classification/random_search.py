import datetime
from scipy.stats import randint, loguniform

import sys
sys.path.insert(1, "../../feature_extractors")
sys.path.insert(1, "../../classification_models")

import dataset_loader
from gru_dwt import GRU_DWT

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
                            "wavelet": wavelet
                        }
                    },
                    "predicted_column": ("power", "apparent"), 
                    "randomsearch": True,
                    "randomsearch_params": {
                        "file_path" : "/home/user/thesis_results/random_search_results/randomsearch_results.csv",
                        "n_iter" : 5,
                        "n_jobs" : -1,
                        "cv" : 5,
                        "model": {
                            "epochs" : randint(1,2),
                            "batch_size" : randint(500,1000),
                            "n_nodes" : randint(32,64),
                        }
                    }
                }),
            },
            "timestep" : 2,
            "train" : {
                "ukdale" : {
                    "location" : "../../../datasets/avEiro_classification/",
                    "houses" : {
                        "house_1" : {
                            "beginning" : datetime.datetime(2020, 10, 1),
                            "end" : datetime.datetime(2020, 10, 2)
                        }
                    }
                },
            },
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
                        experiment[app]["train"][dataset]["houses"],
                        experiment[app]["timestep"]
                    )
            for i in range(0, len(x)):
                X_train.append(x[i])
                y_train.append(y[i])
                
        for method in experiment[app]["methods"]:
            print("Preparing %s" % (method))
            experiment[app]["methods"][method].partial_fit(X_train, [(app, y_train)])

            print("Finished Doing RandomSearch for %s" % (app))

if __name__ == "__main__":
    run_experiment(8, 0, 150, 300, 'bior2.2')
