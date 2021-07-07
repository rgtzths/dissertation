import datetime
import warnings
warnings.filterwarnings("ignore")

from scipy.stats import randint, loguniform

from run_experiment import run

import sys
sys.path.insert(1, "../../classification_models")
sys.path.insert(1, "../../utils")
sys.path.insert(1, "../../feature_extractors")

from seq2point import Seq2Point
from gru import GRU_RNN
from deep_gru import DeepGRU
from lstm import LSTM_RNN
from resnet import ResNet
from mlp_dwt import MLP

base_path= "/home/rteixeira/thesis_results/"
#base_path = "/home/user/thesis_results/"
epochs = 500

def run_experiment():

    #Experiment Definition
    experiment = {
        "heat pump" : {
            "mains_columns" : [('power', 'apparent'), ("voltage", "")],
            "appliance_columns" : [('power', 'apparent')],
            "methods" : {
                "LSTM" : LSTM_RNN({
                    'verbose' : 2,
                    "training_history_folder" : base_path + "history/LSTM/",
                    "results_folder" : base_path + "results/LSTM/",
                    "checkpoint_folder" : base_path + "models/LSTM/",
                    "plots_folder" : base_path + "plots/LSTM/",
                    #"load_model_folder" : base_path + "models/LSTM/",
                    "appliances" : {
                        "heat pump" : {
                            'timewindow' : 180,
                            'timestep' : 2,
                            'overlap' : 178,
                            'epochs' : epochs,
                            'batch_size' : 1024,
                            'n_nodes' : 90,
                        }
                    },
                }),
                "GRU" : GRU_RNN({
                    'verbose' : 2,
                    "training_history_folder" : base_path + "history/GRU/",
                    "results_folder" : base_path + "results/GRU/",
                    "checkpoint_folder" : base_path + "models/GRU/",
                    "plots_folder" : base_path + "plots/GRU/",
                    #"load_model_folder" : base_path + "models/GRU/",
                    "appliances" : {
                        "heat pump" : {
                            'timewindow' : 180,
                            'timestep' : 2,
                            'overlap' : 178,
                            'epochs' : epochs,
                            'batch_size' : 1024,
                            'n_nodes' : 90,
                        }
                    },
                }),
                "DeepGRU" : DeepGRU({
                    'verbose' : 2,
                    "training_history_folder" : base_path + "history/DeepGRU/",
                    "results_folder" : base_path + "results/DeepGRU/",
                    "checkpoint_folder" : base_path + "models/DeepGRU/",
                    "plots_folder" : base_path + "plots/DeepGRU/",
                    #"load_model_folder" : base_path + "models/DeepGRU/",
                    "appliances" : {
                        "heat pump" : {
                            'timewindow' : 180,
                            'timestep' : 2,
                            'overlap' : 178,
                            'epochs' : epochs,
                            'batch_size' : 1024,
                            'n_nodes' : 90
                        }
                    },
                }),
                "ResNet" : ResNet( {
                    "verbose" : 2,
                    "training_history_folder" : base_path + "history/ResNet/",
                    "results_folder" : base_path + "results/ResNet/",
                    "checkpoint_folder" : base_path + "models/ResNet/",
                    "plots_folder" : base_path + "plots/ResNet/",
                    #"load_model_folder" : base_path + "models/ResNet/",
                    "appliances" : {
                        "heat pump" : {
                            'timewindow' : 180,
                            'timestep' : 2,
                            'overlap' : 178,
                            'epochs' : epochs,
                            'batch_size' : 1024,
                            'n_nodes' : 90
                        }
                    },
                }),
                "MLP" : MLP( {
                    "verbose" : 2,
                    "training_history_folder" : base_path + "history/MLP/",
                    "results_folder" : base_path + "results/MLP/",
                    "checkpoint_folder" : base_path + "models/MLP/",
                    "plots_folder" : base_path + "plots/MLP/",
                    #"load_model_folder" : base_path + "models/MLP/",
                    "appliances" : {
                        "heat pump" : {
                            'timewindow' : 180,
                            'timestep' : 2,
                            'overlap' : 178,
                            'epochs' : epochs,
                            'batch_size' : 1024,
                            'feature_extractor' : "dwt"
                        }
                    },
                    "predicted_column": ("power", "apparent"), 
                }),
            },
            "model_path" : base_path + "models/",
            "timestep" : 2,
            "train" : {
                "avEiro" : {
                    "location" : "../../../datasets/avEiro_h5/avEiro.h5",
                    "houses" : {
                        1 : [
                            (datetime.datetime(2020, 10, 5), datetime.datetime(2020, 10, 9)),
                            (datetime.datetime(2020, 10, 20), datetime.datetime(2020, 10, 27)),
                            (datetime.datetime(2020, 11, 1), datetime.datetime(2020, 11, 8)),
                            (datetime.datetime(2020, 12, 9), datetime.datetime(2020, 12, 17)),
                            ]
                        #1 : [(datetime.datetime(2020, 10, 20), datetime.datetime(2020, 10, 21))]
                    }
                },
            },
            "test" : {
                "avEiro" : {
                    "location" : "../../../datasets/avEiro_h5/avEiro.h5",
                    "houses" : {
                        #1 : [ (datetime.datetime(2021, 1, 15), datetime.datetime(2021, 1, 16)) ]
                        1 : [ (datetime.datetime(2021, 1, 15), datetime.datetime(2021, 2, 5)) ]
                    }
                }
            }
        },
        "charger" : {
            "mains_columns" : [('power', 'apparent'), ("voltage", "")],
            "appliance_columns" : [('power', 'apparent')],
            "methods" : {
                "LSTM" : LSTM_RNN({
                    'verbose' : 2,
                    "training_history_folder" : base_path + "history/LSTM/",
                    "results_folder" : base_path + "results/LSTM/",
                    "checkpoint_folder" : base_path + "models/LSTM/",
                    "plots_folder" : base_path + "plots/LSTM/",
                    #"load_model_folder" : base_path + "models/LSTM/",
                    "appliances" : {
                        "charger" : {
                            'timewindow' : 180,
                            'timestep' : 2,
                            'overlap' : 178,
                            'epochs' : epochs,
                            'batch_size' : 1024,
                        }
                    },
                }),
                "GRU" : GRU_RNN({
                    'verbose' : 2,
                    "training_history_folder" : base_path + "history/GRU/",
                    "results_folder" : base_path + "results/GRU/",
                    "checkpoint_folder" : base_path + "models/GRU/",
                    "plots_folder" : base_path + "plots/GRU/",
                    #"load_model_folder" : base_path + "models/GRU/",
                    "appliances" : {
                        "charger" : {
                            'timewindow' : 180,
                            'timestep' : 2,
                            'overlap' : 178,
                            'epochs' : epochs,
                            'batch_size' : 1024,
                        }
                    },
                }),
                "DeepGRU" : DeepGRU({
                    'verbose' : 2,
                    "training_history_folder" : base_path + "history/DeepGRU/",
                    "results_folder" : base_path + "results/DeepGRU/",
                    "checkpoint_folder" : base_path + "models/DeepGRU/",
                    "plots_folder" : base_path + "plots/DeepGRU/",
                    #"load_model_folder" : base_path + "models/DeepGRU/",
                    "appliances" : {
                        "charger" : {
                            'timewindow' : 180,
                            'timestep' : 2,
                            'overlap' : 178,
                            'epochs' : epochs,
                            'batch_size' : 1024,
                            'n_nodes' : 90
                        }
                    },
                }),
                "ResNet" : ResNet( {
                    "verbose" : 2,
                    "training_history_folder" : base_path + "history/ResNet/",
                    "results_folder" : base_path + "results/ResNet/",
                    "checkpoint_folder" : base_path + "models/ResNet/",
                    "plots_folder" : base_path + "plots/ResNet/",
                    #"load_model_folder" : base_path + "models/ResNet/",
                    "appliances" : {
                        "charger" : {
                            'timewindow' : 180,
                            'timestep' : 2,
                            'overlap' : 178,
                            'epochs' : epochs,
                            'batch_size' : 1024,
                            'n_nodes' : 90
                        }
                    },
                }),
                "MLP" : MLP( {
                    "verbose" : 2,
                    "training_history_folder" : base_path + "history/MLP/",
                    "results_folder" : base_path + "results/MLP/",
                    "checkpoint_folder" : base_path + "models/MLP/",
                    "plots_folder" : base_path + "plots/MLP/",
                    #"load_model_folder" : base_path + "models/MLP/",
                    "appliances" : {
                        "charger" : {
                            'timewindow' : 180,
                            'timestep' : 2,
                            'overlap' : 178,
                            'epochs' : epochs,
                            'batch_size' : 1024,
                            'feature_extractor' : "dwt"
                        }
                    },
                }),
            },
            "model_path" : base_path + "models/",
            "timestep" : 2,
            "train" : {
                "avEiro" : {
                    "location" : "../../../datasets/avEiro_h5/avEiro.h5",
                    "houses" : {
                        #1 : [ (datetime.datetime(2020, 11, 14), datetime.datetime(2020, 11, 16)) ]
                        1 : [ 
                            (datetime.datetime(2020, 11, 14), datetime.datetime(2020, 11, 15)),
                            (datetime.datetime(2020, 11, 16), datetime.datetime(2020, 11, 20)),
                            (datetime.datetime(2020, 11, 23), datetime.datetime(2020, 11, 29)),
                            (datetime.datetime(2020, 12, 2), datetime.datetime(2020, 12, 6)),
                            (datetime.datetime(2020, 12, 9), datetime.datetime(2020, 12, 13)),
                            (datetime.datetime(2020, 12, 9), datetime.datetime(2020, 12, 13)),
                            (datetime.datetime(2020, 12, 16), datetime.datetime(2020, 12, 21)),
                            (datetime.datetime(2021, 1, 2), datetime.datetime(2021, 1, 4)),
                            (datetime.datetime(2021, 1, 5), datetime.datetime(2021, 1, 6)),
                            ]
                    }
                },
            },
            "test" : {
                "avEiro" : {
                    "location" : "../../../datasets/avEiro_h5/avEiro.h5",
                    "houses" : {
                        #1 : [ (datetime.datetime(2021, 1, 15), datetime.datetime(2021, 1, 16)) ]
                        1 : [ 
                            (datetime.datetime(2021, 1, 6), datetime.datetime(2021, 1, 9)),
                            (datetime.datetime(2021, 1, 11), datetime.datetime(2021, 1, 21)), 
                            ]
                        
                    }
                }
            }
        }
    }

    run(experiment)

if __name__ == "__main__":
    run_experiment()
