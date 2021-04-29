import datetime
import warnings
warnings.filterwarnings("ignore")

from scipy.stats import randint, loguniform

from run_experiment import run

import sys
sys.path.insert(1, "../../classification_models")
from gacd import GACD
from seq2point import Seq2Point
from gru_dwt import GRU_DWT

def run_experiment():

    #Experiment Definition
    experiment = {
        "heatpump" : {
            "mains_columns" : [('power', 'apparent')],
            "appliance_columns" : [('power', 'apparent')],
            "methods" : {
                "GACD" :GACD( {
                    "verbose" : 2,
                    "training_results_path" : "/home/user/thesis_results/history/",
                    "appliances" : {
                        "heatpump" : {
                            "dwt_timewindow" : 12,
                            "dwt_overlap" : 10,
                            "examples_overlap" : 150,
                            "examples_timewindow" : 300,
                            "epochs" : 1,
                            "batch_size" : 1024,
                            "wavelet": 'bior2.2',
                        }
                    },
                    "predicted_column": ("power", "apparent"), 
                }),
                "GRU_DWT" : GRU_DWT( {
                    "verbose" : 2,
                    "training_results_path" : "/home/user/thesis_results/history/",
                    "appliances" : {
                        "heatpump" : {
                            "dwt_timewindow" : 12,
                            "dwt_overlap" : 10,
                            "examples_overlap" : 150,
                            "examples_timewindow" : 300,
                            "epochs" : 1,
                            "batch_size" : 1024,
                            "wavelet": 'bior2.2',
                        }
                    },
                    "predicted_column": ("power", "apparent"), 
                }),
                "Seq2Point" : Seq2Point({'n_epochs':1,'batch_size':1024, "training_results_path" : "/home/user/thesis_results/history/"})
            },
            "model_path" : "/home/user/thesis_results/models/",
            "timestep" : 2,
            "train" : {
                "avEiro" : {
                    "location" : "../../../datasets/avEiro_classification/",
                    "houses" : {
                        "house_1" : {
                            "beginning" : datetime.datetime(2020, 10, 1),
                            #"end" : datetime.datetime(2021, 1, 15)
                            "end" : datetime.datetime(2020, 10, 2)
                        }
                    }
                },
            },
            "test" : {
                "avEiro" : {
                    "location" : "../../../datasets/avEiro_classification/",
                    "houses" : {
                        "house_1" : {
                            "beginning" : datetime.datetime(2021, 1, 15),
                            #"end" : datetime.datetime(2021, 1, 25)
                            "end" : datetime.datetime(2021, 1, 15, 12)

                        }
                    }
                }
            }
        },
        "carcharger" : {
            "mains_columns" : [('power', 'apparent')],
            "appliance_columns" : [('power', 'apparent')],
            "methods" : {
                "GACD" :GACD( {
                    "verbose" : 2,
                    "training_results_path" : "/home/user/thesis_results/history/",
                    "appliances" : {
                        "carcharger" : {
                            "dwt_timewindow" : 12,
                            "dwt_overlap" : 10,
                            "examples_overlap" : 150,
                            "examples_timewindow" : 300,
                            "epochs" : 1,
                            "batch_size" : 1024,
                            "wavelet": 'bior2.2',
                        }
                    },
                    "predicted_column": ("power", "apparent"), 
                }),
                "GRU_DWT" : GRU_DWT( {
                    "verbose" : 2,
                    "training_results_path" : "/home/user/thesis_results/history/",
                    "appliances" : {
                        "carcharger" : {
                            "dwt_timewindow" : 12,
                            "dwt_overlap" : 10,
                            "examples_overlap" : 150,
                            "examples_timewindow" : 300,
                            "epochs" : 1,
                            "batch_size" : 1024,
                            "wavelet": 'bior2.2',
                        }
                    },
                    "predicted_column": ("power", "apparent"), 
                }),
                "Seq2Point" : Seq2Point({'n_epochs':1,'batch_size':1024, "training_results_path" : "/home/user/thesis_results/history/"})
            },
            "model_path" : "/home/user/thesis_results/models/",
            "timestep" : 2,
            "train" : {
                "ukdale" : {
                    "location" : "../../../datasets/avEiro_classification/",
                    "houses" : {
                        "house_1" : {
                            "beginning" : datetime.datetime(2020, 11, 14),
                            #"end" : datetime.datetime(2021, 1, 25)
                            "end" : datetime.datetime(2020, 11, 15)
                        }
                    }
                },
            },
            "test" : {
                "ukdale" : {
                    "location" : "../../../datasets/avEiro_classification/",
                    "houses" : {
                        "house_1" : {
                            "beginning" : datetime.datetime(2021, 1, 25),
                            #"end" : datetime.datetime(2021, 2, 6)
                            "end" : datetime.datetime(2021, 1, 25, 12)
                        }
                    }
                }
            }
        }
    }

    run(experiment)

if __name__ == "__main__":
    run_experiment()
