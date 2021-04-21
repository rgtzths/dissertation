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
            },
            "model_path" : "./models/",
            "timestep" : 2,
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
