import datetime
import warnings
warnings.filterwarnings("ignore")

from scipy.stats import randint, loguniform

from run_experiment import run

from lstm import LSTM_RNN
from gru import GRU_RNN
from gradient import GradientBoosting
from cnn import CNN
from svm import SVM
from gru_dwt import GRU_DWT

import sys
import logging

#sys.stderr = open('./outputs/err.log', 'w')

#logging.basicConfig(level=logging.INFO,
#                    format='%(asctime)s %(levelname)s %(message)s',
#                    filename='./outputs/info.log',
#                    filemode='w')


def run_experiment():

    #Experiment Definition
    experiment = {
        "heatpump" : {
            "methods" : {
                "GRU_DWT" :GRU_DWT( {
                    "verbose" : 2,
                    "appliances" : {
                        "heatpump" : {
                            "dwt_timewindow" : 12,
                            "timestep" : 2,
                            "dwt_overlap" : 6,
                            "examples_overlap" : 150,
                            "examples_timewindow" : 300,
                            "epochs" : 1,
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
                        "n_iter" : 3,
                        "n_jobs" : -1,
                        "cv" : 5,
                        "model": {
                            "epochs" : randint(1,2),
                            "batch_size" : randint(500,1000),
                        }
                    }
                }),
                #"LSTM" : LSTM( {
                #    "verbose" : 2,
                #    "timewindow": {"heatpump" : 5}, 
                #    "timestep": {"heatpump" : 2},
                #    "n_nodes" : {"heatpump" : 150},
                #    "epochs" : {"heatpump" : 300},
                #    "batch_size" : {"heatpump" : 1000},
                #    "predicted_column": ("power", "apparent"), 
                #    "randomsearch": False,
                #    "randomsearch_params": {
                #        "timewindow" : [1],
                #        "timestep" : [2],
                #        "n_iter" : 1,
                #        "n_jobs" : -1,
                #        "cv" : 5,
                #        "model": {
                #            "epochs": randint(10,50),
                #            "batch_size" : randint(10,1000),
                #        }
                #    }
                #}),
                #"GRU" : GRU( {
                #    "verbose" : 2,
                #    "timewindow": {"heatpump" : 5}, 
                #    "timestep": {"heatpump" : 2},
                #    "n_nodes" : {"heatpump" : 150},
                #    "epochs" : {"heatpump" : 300},
                #    "batch_size" : {"heatpump" : 1000},
                #    "predicted_column": ("power", "apparent"),  
                #    "randomsearch": False,
                #    "randomsearch_params": {
                #        "timewindow" : [1],
                #        "timestep" : [2],
                #        "n_iter" : 1,
                #        "n_jobs" : -1,
                #        "cv" : 5,
                #        "model": {
                #            "epochs": randint(10,50),
                #            "batch_size" : randint(10,1000),
                #        }
                #    }
                #}),
                #"Gradient" : GradientBoosting({
                #    "verbose" : 2,
                #    "timewindow": {"heatpump" : 5}, 
                #    "timestep": {"heatpump" : 2}, 
                #    "predicted_column": ("power", "apparent"), 
                #    "params" : {
                #        "heatpump" : {
                #            'n_estimators': 500,
                #            'max_depth': 4,
                #            'min_samples_split': 5,
                #            'learning_rate': 0.01,
                #            'loss': 'deviance'
                #        }
                #    },
                #    "randomsearch" : False,
                #    "randomsearch_params" : {
                #        "timewindow" : [1],
                #        "timestep" : [2],
                #        "n_iter" : 1,
                #        "n_jobs" : -1,
                #        "cv" : 5,
                #        "model": {
                #            'n_estimators': randint(200,1000),
                #            'max_depth': randint(2,10),
                #            'min_samples_split': randint(2,10),
                #            'learning_rate': loguniform(0.001, 1),
                #            'loss': ['deviance']
                #        }
                #    }
                #}),
                #"CNN" : CNN( {
                #    "verbose" : 2,
                #    "timewindow": {"heatpump" : 1}, 
                #    "timestep": {"heatpump" : 2},
                #    "epochs" : {"heatpump" : 10},
                #    "batch_size" : {"heatpump" : 500},
                #    "predicted_column": ("power", "apparent"), 
                #    "randomsearch": False,
                #    "randomsearch_params": {
                #        "timewindow" : [1],
                #        "timestep" : [2],
                #        "cv" : 5,
                #        "n_iter" : 1,
                #        "n_jobs" : 1,
                #        "model": {
                #            "epochs": randint(1,10),
                #            "batch_size" : randint(10,1000),
                #        }
                #    }
                #}),
                #"SVM" : SVM({
                #    "verbose" : 2,
                #    "timewindow": {"heatpump" : 1}, 
                #    "timestep": {"heatpump" : 2}, 
                #    "predicted_column": ("power", "apparent"), 
                #    "params" : {
                #        "heatpump" : {
                #            'kernel': 'rbf',
                #            'C': 0.1,
                #            'degree': 1,
                #            'coef0': 0.1,
                #            'tol' : 0.0001
                #        }
                #    },
                #    "randomsearch" : False,
                #    "randomsearch_params" : {
                #        "timewindow" : [1],
                #        "timestep" : [2],
                #        "n_iter" : 1,
                #        "n_jobs" : -1,
                #        "cv" : 5,
                #        "model": {
                #            'kernel': ['rbf', 'poly'],
                #            'C': loguniform(0.001, 1),
                #            'degree': randint(2,10),
                #            'coef0': loguniform(0.001, 1),
                #            'tol' : loguniform(0.00001, 0.01)
                #        }
                #    }
                #})
            },
            "model_path" : "./models/",
            "train" : {
                "ukdale" : {
                    "location" : "../../datasets/avEiro_classification/",
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

    run(experiment)

if __name__ == "__main__":
    run_experiment()
