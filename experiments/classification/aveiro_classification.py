import datetime
import warnings
warnings.filterwarnings("ignore")

from scipy.stats import randint, loguniform

from run_experiment import run

import sys
sys.path.insert(1, "../../classification_models")
from gru_dwt import GRU_DWT

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
                            "dwt_overlap" : 0,
                            "examples_overlap" : 0,
                            "examples_timewindow" : 300,
                            "epochs" : 1,
                            "batch_size" : 1024,
                            "wavelet": 'bior2.2'
                        }
                    },
                    "predicted_column": ("power", "apparent"), 
                }),
            },
            "model_path" : "~/thesis_results/models/",
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
            "test" : {
                "ukdale" : {
                    "location" : "../../../datasets/avEiro_classification/",
                    "houses" : {
                        "house_1" : {
                            "beginning" : datetime.datetime(2021, 1, 15),
                            "end" : datetime.datetime(2021, 1, 15, 12)
                        }
                    }
                }
            }
        }
    }

    run(experiment)

if __name__ == "__main__":
    run_experiment()
