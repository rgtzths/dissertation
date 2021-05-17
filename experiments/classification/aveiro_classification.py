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
from gru import GRU_RNN
from lstm import LSTM_RNN
from resnet import ResNet

def run_experiment():

    #Experiment Definition
    experiment = {
        "heatpump" : {
            "mains_columns" : [('power', 'apparent'), ("voltage", "")],
            "appliance_columns" : [('power', 'apparent')],
            "methods" : {
                "GACD" :GACD( {
                    "verbose" : 2,
                    "training_results_path" : "/home/user/thesis_results/history/",
                    "results_path" : "/home/user/thesis_results/results/GACD/results_heatpump.txt",
                    "checkpoint_file" : "/home/user/thesis_results/models/GACD/model_checkpoint_heatpump.h5",
                    "load_model_folder" : "/home/user/thesis_results/models/GACD/",
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
                    "results_path" : "/home/user/thesis_results/results/GRU_DWT/results_heatpump.txt",
                    "checkpoint_file" : "/home/user/thesis_results/models/GRU_DWT/model_checkpoint_heatpump.h5",
                    "load_model_folder" : "/home/user/thesis_results/models/GRU_DWT/",
                    "appliances" : {
                        "heatpump" : {
                            "dwt_timewindow" : 12,
                            "dwt_overlap" : 10,
                            "examples_overlap" : 60,
                            "examples_timewindow" : 120,
                            "epochs" : 1,
                            "batch_size" : 1024,
                            "wavelet": 'db4',
                        }
                    },
                    "predicted_column": ("power", "apparent"), 
                }),
                "LSTM" : LSTM_RNN({
                    'timewindow' : 180,
                    'timestep' : 2,
                    'overlap' : 178,
                    'verbose' : 2,
                    'n_nodes' : 90,
                    'epochs' : 5,
                    'batch_size' : 1024,
                    "training_results_path" : "/home/user/thesis_results/history/",
                    "results_path" : "/home/user/thesis_results/results/LSTM/results_heatpump.txt",
                    "checkpoint_file" : "/home/user/thesis_results/models/LSTM/model_checkpoint_heatpump.h5",
                    "load_model_folder" : "/home/user/thesis_results/models/LSTM/",
                }),
                "GRU" : GRU_RNN({
                    'timewindow' : 180,
                    'timestep' : 2,
                    'overlap' : 178,
                    'verbose' : 2,
                    'n_nodes' : 90,
                    'epochs' : 1,
                    'batch_size' : 1024,
                    "training_results_path" : "/home/user/thesis_results/history/",
                    "results_path" : "/home/user/thesis_results/results/GRU/results_heatpump.txt",
                    "checkpoint_file" : "/home/user/thesis_results/models/GRU/model_checkpoint_heatpump.h5",
                    "load_model_folder" : "/home/user/thesis_results/models/GRU/",
                }),
                "ResNet" : ResNet( {
                    "verbose" : 2,
                    "training_results_path" : "/home/user/thesis_results/history/",
                    "results_path" : "/home/user/thesis_results/results/ResNet/results_heatpump.txt",
                    "checkpoint_file" : "/home/user/thesis_results/models/ResNet/model_checkpoint_heatpump.h5",
                    #"load_model_folder" : "/home/user/thesis_results/models/ResNet/",
                    "appliances" : {
                        "heatpump" : {
                            'timewindow' : 180,
                            'timestep' : 2,
                            'overlap' : 178,
                            'epochs' : 1,
                            'batch_size' : 1024,
                        }
                    },
                    "predicted_column": ("power", "apparent"), 
                }),
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
                            #"end" : datetime.datetime(2020, 12, 2)
                            "end" : datetime.datetime(2020, 10, 3)
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
                            #"end" : datetime.datetime(2021, 2, 6)
                            #"end" : datetime.datetime(2021, 1, 20)
                            "end" : datetime.datetime(2021, 1, 16)

                        }
                    }
                }
            }
        },
        "carcharger" : {
            "mains_columns" : [('power', 'apparent'), ("voltage", "")],
            "appliance_columns" : [('power', 'apparent')],
            "methods" : {
                "GACD" :GACD( {
                    "verbose" : 2,
                    "training_results_path" : "/home/user/thesis_results/history/",
                    "results_path" : "/home/user/thesis_results/results/GACD/results_carcharger.txt",
                    "checkpoint_file" : "/home/user/thesis_results/models/GACD/model_checkpoint_carcharger.h5",
                    #"load_model_folder" : "/home/user/thesis_results/models/GACD/",
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
                    "results_path" : "/home/user/thesis_results/results/GRU_DWT/results_carcharger.txt",
                    "checkpoint_file" : "/home/user/thesis_results/models/GRU_DWT/model_checkpoint_carcharger.h5",
                    #"load_model_folder" : "/home/user/thesis_results/models/GRU_DWT/",
                    "appliances" : {
                        "carcharger" : {
                            "dwt_timewindow" : 12,
                            "dwt_overlap" : 10,
                            "examples_overlap" : 60,
                            "examples_timewindow" : 120,
                            "epochs" : 1,
                            "batch_size" : 1024,
                            "wavelet": 'db4',
                        }
                    },
                    "predicted_column": ("power", "apparent"), 
                }),
                "LSTM" : LSTM_RNN({
                    'timewindow' : 180,
                    'timestep' : 2,
                    'overlap' : 178,
                    'verbose' : 2,
                    'n_nodes' : 90,
                    'epochs' : 1,
                    'batch_size' : 1024,
                    "training_results_path" : "/home/user/thesis_results/history/",
                    "results_path" : "/home/user/thesis_results/results/LSTM/results_carcharger.txt",
                    "checkpoint_file" : "/home/user/thesis_results/models/LSTM/model_checkpoint_carcharger.h5",
                    #"load_model_folder" : "/home/user/thesis_results/models/LSTM/",
                }),
                "GRU" : GRU_RNN({
                    'timewindow' : 180,
                    'timestep' : 2,
                    'overlap' : 178,
                    'verbose' : 2,
                    'n_nodes' : 90,
                    'epochs' : 1,
                    'batch_size' : 1024,
                    "training_results_path" : "/home/user/thesis_results/history/",
                    "results_path" : "/home/user/thesis_results/results/GRU/results_carcharger.txt",
                    "checkpoint_file" : "/home/user/thesis_results/models/GRU/model_checkpoint_carcharger.h5",
                    #"load_model_folder" : "/home/user/thesis_results/models/GRU/",
                }),
                "ResNet" : ResNet( {
                    "verbose" : 2,
                    "training_results_path" : "/home/user/thesis_results/history/",
                    "results_path" : "/home/user/thesis_results/results/ResNet/results_carcharger.txt",
                    "checkpoint_file" : "/home/user/thesis_results/models/ResNet/model_checkpoint_carcharger.h5",
                    #"load_model_folder" : "/home/user/thesis_results/models/ResNet/",
                    "appliances" : {
                        "carcharger" : {
                            'timewindow' : 180,
                            'timestep' : 2,
                            'overlap' : 178,
                            'epochs' : 1,
                            'batch_size' : 1024,
                        }
                    },
                    "predicted_column": ("power", "apparent"), 
                }),
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
                            "end" : datetime.datetime(2020, 11, 16)
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
                            #"end" : datetime.datetime(2021, 2, 1)
                            "end" : datetime.datetime(2021, 1, 26)
                        }
                    }
                }
            }
        }
    }

    run(experiment)

if __name__ == "__main__":
    run_experiment()
