import datetime
import warnings
warnings.filterwarnings("ignore")

from scipy.stats import randint, loguniform

from run_experiment import run

import sys
sys.path.insert(1, "../../classification_models")

from seq2point import Seq2Point
from gru import GRU_RNN
from gru2 import GRU2
from lstm import LSTM_RNN
from resnet import ResNet
from mlp_dwt import MLP

#base_path= "/home/rteixeira/thesis_results/"
base_path = "/home/user/thesis_results/"
epochs = 1

def run_experiment():

    #Experiment Definition
    experiment = {
        #"heatpump" : {
        #    "mains_columns" : [('power', 'apparent'), ("voltage", "")],
        #    "appliance_columns" : [('power', 'apparent')],
        #    "methods" : {
        #        "LSTM" : LSTM_RNN({
        #            'verbose' : 2,
        #            "training_results_path" : base_path + "history/",
        #            "results_path" : base_path + "results/LSTM/results_heatpump.txt",
        #            "checkpoint_file" : base_path + "models/LSTM/model_checkpoint_heatpump.h5",
        #            "load_model_folder" : base_path + "models/LSTM/",
        #            "appliances" : {
        #                "heatpump" : {
        #                    'timewindow' : 180,
        #                    'timestep' : 2,
        #                    'overlap' : 178,
        #                    'epochs' : epochs,
        #                    'batch_size' : 1024,
        #                    'n_nodes' : 90,
        #                }
        #            },
        #        }),
        #        "GRU" : GRU_RNN({
        #            'verbose' : 2,
        #            "training_results_path" : base_path + "history/",
        #            "results_path" : base_path + "results/GRU/results_heatpump.txt",
        #            "checkpoint_file" : base_path + "models/GRU/model_checkpoint_heatpump.h5",
        #            "load_model_folder" : base_path + "models/GRU/",
        #            "appliances" : {
        #                "heatpump" : {
        #                    'timewindow' : 180,
        #                    'timestep' : 2,
        #                    'overlap' : 178,
        #                    'epochs' : epochs,
        #                    'batch_size' : 1024,
        #                    'n_nodes' : 90,
        #                }
        #            },
        #        }),
        #        "GRU2" : GRU2({
        #            'verbose' : 2,
        #            "training_results_path" : base_path + "history/",
        #            "results_path" : base_path + "results/GRU2/results_heatpump.txt",
        #            "checkpoint_file" : base_path + "models/GRU2/model_checkpoint_heatpump.h5",
        #            "load_model_folder" : base_path + "models/GRU2/",
        #            "appliances" : {
        #                "heatpump" : {
        #                    'timewindow' : 180,
        #                    'timestep' : 2,
        #                    'overlap' : 178,
        #                    'epochs' : epochs,
        #                    'batch_size' : 1024,
        #                    'n_nodes' : 90,
        #                }
        #            },
        #        }),
        #        "ResNet" : ResNet( {
        #            "verbose" : 2,
        #            "training_results_path" : base_path + "history/",
        #            "results_path" : base_path + "results/ResNet/results_heatpump.txt",
        #            "checkpoint_file" : base_path + "models/ResNet/model_checkpoint_heatpump.h5",
        #            "load_model_folder" : base_path + "models/ResNet/",
        #            "appliances" : {
        #                "heatpump" : {
        #                    'timewindow' : 180,
        #                    'timestep' : 2,
        #                    'overlap' : 178,
        #                    'epochs' : epochs,
        #                    'batch_size' : 1024,
        #                }
        #            },
        #        }),
        #        "MLP" : MLP( {
        #            "verbose" : 2,
        #            "training_results_path" : base_path + "history/",
        #            "results_path" : base_path + "results/MLP/results_heatpump.txt",
        #            "checkpoint_file" : base_path + "models/MLP/model_checkpoint_heatpump.h5",
        #            "load_model_folder" : base_path + "models/MLP/",
        #            "appliances" : {
        #                "heatpump" : {
        #                    'timewindow' : 180,
        #                    'timestep' : 2,
        #                    'overlap' : 172,
        #                    'epochs' : epochs,
        #                    'batch_size' : 1024,
        #                    'feature_extractor' : "dwt"
        #                }
        #            },
        #            "predicted_column": ("power", "apparent"), 
        #        }),
        #    },
        #    "model_path" : base_path + "models/",
        #    "timestep" : 2,
        #    "train" : {
        #        "avEiro" : {
        #            "location" : "../../../datasets/avEiro_classification/",
        #            "houses" : {
        #                "house_1" : [
        #                    (datetime.datetime(2020, 10, 20), datetime.datetime(2020, 10, 27)),
        #                    (datetime.datetime(2020, 11, 1), datetime.datetime(2020, 11, 8)),
        #                    (datetime.datetime(2020, 12, 9), datetime.datetime(2020, 12, 17)),
        #                    (datetime.datetime(2021, 1, 17), datetime.datetime(2021, 1, 25))
        #                    ]
        #                #"house_1" : [(datetime.datetime(2020, 10, 20), datetime.datetime(2020, 10, 21)), (datetime.datetime(2020, 10, 21), datetime.datetime(2020, 10, 22))]
        #                #"house_1" : [(datetime.datetime(2020, 10, 20), datetime.datetime(2020, 10, 25))]
        #            }
        #        },
        #    },
        #    "test" : {
        #        "avEiro" : {
        #            "location" : "../../../datasets/avEiro_classification/",
        #            "houses" : {
        #                "house_1" : [ (datetime.datetime(2021, 1, 15), datetime.datetime(2021, 2, 5)) ]
        #            }
        #        }
        #    }
        #},
        "carcharger" : {
            "mains_columns" : [('power', 'apparent'), ("voltage", "")],
            "appliance_columns" : [('power', 'apparent')],
            "methods" : {
                "LSTM" : LSTM_RNN({
                    'verbose' : 2,
                    "training_results_path" : base_path + "history/",
                    "results_path" : base_path + "results/LSTM/results_carcharger.txt",
                    "checkpoint_file" : base_path + "models/LSTM/model_checkpoint_carcharger.h5",
                    #"load_model_folder" : base_path + "models/LSTM/",
                    "appliances" : {
                        "carcharger" : {
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
                    "training_results_path" : base_path + "history/",
                    "results_path" : base_path + "results/GRU/results_carcharger.txt",
                    "checkpoint_file" : base_path + "models/GRU/model_checkpoint_carcharger.h5",
                    #"load_model_folder" : base_path + "models/GRU/",
                    "appliances" : {
                        "carcharger" : {
                            'timewindow' : 180,
                            'timestep' : 2,
                            'overlap' : 178,
                            'epochs' : epochs,
                            'batch_size' : 1024,
                        }
                    },
                }),
                "GRU2" : GRU2({
                    'verbose' : 2,
                    "training_results_path" : base_path + "history/",
                    "results_path" : base_path + "results/GRU2/results_carcharger.txt",
                    "checkpoint_file" : base_path + "models/GRU2/model_checkpoint_carcharger.h5",
                    #"load_model_folder" : base_path + "models/GRU2/",
                    "appliances" : {
                        "carcharger" : {
                            'timewindow' : 180,
                            'timestep' : 2,
                            'overlap' : 178,
                            'epochs' : epochs,
                            'batch_size' : 1024,
                        }
                    },
                }),
                "ResNet" : ResNet( {
                    "verbose" : 2,
                    "training_results_path" : base_path + "history/",
                    "results_path" : base_path + "results/ResNet/results_carcharger.txt",
                    "checkpoint_file" : base_path + "models/ResNet/model_checkpoint_carcharger.h5",
                    #"load_model_folder" : base_path + "models/ResNet/",
                    "appliances" : {
                        "carcharger" : {
                            'timewindow' : 180,
                            'timestep' : 2,
                            'overlap' : 178,
                            'epochs' : epochs,
                            'batch_size' : 1024,
                            'n_nodes' : 90,
                        }
                    },
                }),
                "MLP" : MLP( {
                    "verbose" : 2,
                    "training_results_path" : base_path + "history/",
                    "results_path" : base_path + "results/MLP/results_carcharger.txt",
                    "checkpoint_file" : base_path + "models/MLP/model_checkpoint_carcharger.h5",
                    #"load_model_folder" : base_path + "models/MLP/",
                    "appliances" : {
                        "carcharger" : {
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
                "ukdale" : {
                    "location" : "../../../datasets/avEiro_classification/",
                    "houses" : {
                        "house_1" : [ (datetime.datetime(2020, 11, 14), datetime.datetime(2020, 11, 16)) ]
                        #"house_1" : [ 
                        #    (datetime.datetime(2020, 11, 14), datetime.datetime(2020, 11, 22)),
                        #    (datetime.datetime(2020, 12, 1), datetime.datetime(2020, 12, 15)),
                        #    (datetime.datetime(2021, 1, 1), datetime.datetime(2021, 1, 8)) 
                        #    ]
                    }
                },
            },
            "test" : {
                "ukdale" : {
                    "location" : "../../../datasets/avEiro_classification/",
                    "houses" : {
                        "house_1" : [ (datetime.datetime(2021, 1, 15), datetime.datetime(2021, 1, 16)) ]
                        #"house_1" : [ (datetime.datetime(2021, 1, 15), datetime.datetime(2021, 2, 5)) ]
                    }
                }
            }
        }
    }

    run(experiment)

if __name__ == "__main__":
    run_experiment()
