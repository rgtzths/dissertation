import datetime
import warnings
warnings.filterwarnings("ignore")

from run_experiment import run

import sys
sys.path.insert(1, "../../classification_models")
sys.path.insert(1, "../../utils")
sys.path.insert(1, "../../feature_extractors")

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
        "fridge" : {
            "mains_columns" : [('power', 'apparent')],
            "appliance_columns" : [('power', 'active')],
            "methods" : {
                "LSTM" : LSTM_RNN({
                    'verbose' : 2,
                    "training_history_folder" : base_path + "history/LSTM/",
                    "results_folder" : base_path + "results/LSTM/",
                    "checkpoint_folder" : base_path + "models/LSTM/",
                    "plots_folder" : base_path + "plots/LSTM/",
                    #"load_model_folder" : base_path+"models/LSTM/",
                    "appliances" : {
                        "fridge" : {
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
                    #"load_model_folder" : base_path+"models/GRU/",
                    "appliances" : {
                        "fridge" : {
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
                    #"load_model_folder" : base_path+"models/DeepGRU/",
                    "appliances" : {
                        "fridge" : {
                            'timewindow' : 180,
                            'timestep' : 2,
                            'overlap' : 178,
                            'epochs' : epochs,
                            'batch_size' : 1024,
                            'n_nodes' : 90,
                        }
                    },
                }),
                "ResNet" : ResNet( {
                    "verbose" : 2,
                   "training_history_folder" : base_path + "history/ResNet/",
                    "results_folder" : base_path + "results/ResNet/",
                    "checkpoint_folder" : base_path + "models/ResNet/",
                    "plots_folder" : base_path + "plots/ResNet/",
                    #"load_model_folder" : base_path+"models/ResNet/",
                    "appliances" : {
                        "fridge" : {
                            'timewindow' : 180,
                            'timestep' : 2,
                            'overlap' : 178,
                            'epochs' : epochs,
                            'batch_size' : 1024,
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
                        "fridge" : {
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
            "model_path" : base_path+"models/",
            "timestep" : 6,
            "train" : {
                "ukdale" : {
                    "location" : "../../../datasets/ukdale/ukdale.h5",
                    "houses" : {
                        #1 : [(datetime.datetime(2014, 2, 22), datetime.datetime(2014, 2, 22, 23, 59, 53))],
                        #2 : [(datetime.datetime(2013, 5, 23), datetime.datetime(2013, 5, 24))]
                        1 : [
                            (datetime.datetime(2014, 2, 20), datetime.datetime(2014, 2, 28)),
                             (datetime.datetime(2016, 9, 10), datetime.datetime(2016, 9, 17))
                            ],
                        2 : [
                            (datetime.datetime(2013, 5, 22), datetime.datetime(2013, 5, 29)),
                            (datetime.datetime(2013, 6, 7), datetime.datetime(2013, 7, 14))
                            ]
                    }
                },
            },
            "test" : {
                "ukdale" : {
                    "location" : "../../../datasets/ukdale/ukdale.h5",
                    "houses" : {
                        #5 : [(datetime.datetime(2014, 8, 1),datetime.datetime(2014, 8, 2))]
                        5 : [
                            (datetime.datetime(2014, 6, 30), datetime.datetime(2014, 7, 15)),
                            ]
                    }
                }
            }
        },
        "microwave" : {
            "mains_columns" : [('power', 'apparent')],
            "appliance_columns" : [('power', 'active')],
            "methods" : {
                "LSTM" : LSTM_RNN({
                    'verbose' : 2,
                     "training_history_folder" : base_path + "history/LSTM/",
                    "results_folder" : base_path + "results/LSTM/",
                    "checkpoint_folder" : base_path + "models/LSTM/",
                    "plots_folder" : base_path + "plots/LSTM/",
                    #"load_model_folder" : base_path+"models/LSTM/",
                    "appliances" : {
                        "microwave" : {
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
                    #"load_model_folder" : base_path+"models/GRU/",
                    "appliances" : {
                        "microwave" : {
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
                    #"load_model_folder" : base_path+"models/DeepGRU/",
                    "appliances" : {
                        "microwave" : {
                            'timewindow' : 180,
                            'timestep' : 2,
                            'overlap' : 178,
                            'epochs' : epochs,
                            'batch_size' : 1024,
                            'n_nodes' : 90,
                        }
                    },
                }),
                "ResNet" : ResNet( {
                    "verbose" : 2,
                    "training_history_folder" : base_path + "history/ResNet/",
                    "results_folder" : base_path + "results/ResNet/",
                    "checkpoint_folder" : base_path + "models/ResNet/",
                    "plots_folder" : base_path + "plots/ResNet/",
                    #"load_model_folder" : base_path+"models/ResNet/",
                    "appliances" : {
                        "microwave" : {
                            'timewindow' : 180,
                            'timestep' : 2,
                            'overlap' : 178,
                            'epochs' : epochs,
                            'batch_size' : 1024,
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
                        "microwave" : {
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
            "model_path" : base_path+"models/",
            "timestep" : 6,
            "train" : {
                "ukdale" : {
                    "location" : "../../../datasets/ukdale/ukdale.h5",
                    "houses" : {
                        #1 : [(datetime.datetime(2014, 2, 22), datetime.datetime(2014, 2, 23, 23, 59, 48))],
                        #2 : [(datetime.datetime(2013, 5, 23), datetime.datetime(2013, 5, 24))]
                        1 : [
                            (datetime.datetime(2013, 7, 12), datetime.datetime(2013, 7, 18)),
                            (datetime.datetime(2013, 7, 28), datetime.datetime(2013, 7, 5)),
                            ],
                        2 : [
                            (datetime.datetime(2013, 6, 5), datetime.datetime(2013, 6, 12)),
                            (datetime.datetime(2013, 7, 18), datetime.datetime(2013, 7, 25)),
                            ]
                    }
                },
            },
            "test" : {
                "ukdale" : {
                    "location" : "../../../datasets/ukdale/ukdale.h5",
                    "houses" : {
                        #5 : [(datetime.datetime(2014, 8, 1),datetime.datetime(2014, 8, 2))]
                        5 : [
                            (datetime.datetime(2014, 7, 8), datetime.datetime(2014, 7, 9)),
                            (datetime.datetime(2014, 7, 10), datetime.datetime(2014, 7, 11)),
                            (datetime.datetime(2014, 7, 12), datetime.datetime(2014, 7, 17)),
                            (datetime.datetime(2014, 7, 21), datetime.datetime(2014, 7, 22)),
                            (datetime.datetime(2014, 7, 28), datetime.datetime(2014, 7, 29)),
                            (datetime.datetime(2014, 8, 4), datetime.datetime(2014, 8, 5)),
                            (datetime.datetime(2014, 8, 7), datetime.datetime(2014, 8, 8)),
                            (datetime.datetime(2014, 8, 10), datetime.datetime(2014, 8, 12)),
                            ]
                    }
                }
            }
        },
        "dish washer" : {
            "mains_columns" : [('power', 'apparent')],
            "appliance_columns" : [('power', 'active')],
            "methods" : {
                "LSTM" : LSTM_RNN({
                    'verbose' : 2,
                    "training_history_folder" : base_path + "history/LSTM/",
                    "results_folder" : base_path + "results/LSTM/",
                    "checkpoint_folder" : base_path + "models/LSTM/",
                    "plots_folder" : base_path + "plots/LSTM/",
                    #"load_model_folder" : base_path+"models/LSTM/",
                    "appliances" : {
                        "dish_washer" : {
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
                    #"load_model_folder" : base_path+"models/GRU/",
                    "appliances" : {
                        "dish_washer" : {
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
                    #"load_model_folder" : base_path+"models/DeepGRU/",
                    "appliances" : {
                        "dish_washer" : {
                            'timewindow' : 180,
                            'timestep' : 2,
                            'overlap' : 178,
                            'epochs' : epochs,
                            'batch_size' : 1024,
                            'n_nodes' : 90,
                        }
                    },
                }),
                "ResNet" : ResNet( {
                    "verbose" : 2,
                    "training_history_folder" : base_path + "history/ResNet/",
                    "results_folder" : base_path + "results/ResNet/",
                    "checkpoint_folder" : base_path + "models/ResNet/",
                    "plots_folder" : base_path + "plots/ResNet/",
                    #"load_model_folder" : base_path+"models/ResNet/",
                    "appliances" : {
                        "dish_washer" : {
                            'timewindow' : 180,
                            'timestep' : 2,
                            'overlap' : 178,
                            'epochs' : epochs,
                            'batch_size' : 1024,
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
                        "dish_washer" : {
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
            "model_path" : base_path+"models/",
            "timestep" : 6,
            "train" : {
                "ukdale" : {
                    "location" : "../../../datasets/ukdale/ukdale.h5",
                    "houses" : {
                        #1 : [(datetime.datetime(2014, 2, 22), datetime.datetime(2014, 2, 23, 23, 59, 50))],
                        #2 : [(datetime.datetime(2015, 9, 23), datetime.datetime(2015, 9, 24))]
                        1 : [
                            (datetime.datetime(2013, 2, 18), datetime.datetime(2013, 2, 19)),
                            (datetime.datetime(2013, 2, 21), datetime.datetime(2013, 2, 22)),
                            (datetime.datetime(2013, 2, 24), datetime.datetime(2013, 2, 25)),
                            (datetime.datetime(2013, 2, 27), datetime.datetime(2013, 2, 28)),
                            (datetime.datetime(2013, 3, 9), datetime.datetime(2013, 3, 10)),
                            (datetime.datetime(2013, 3, 16), datetime.datetime(2013, 3, 17)),
                            (datetime.datetime(2013, 3, 19), datetime.datetime(2013, 3, 20)),
                            (datetime.datetime(2013, 3, 21), datetime.datetime(2013, 3, 22)),
                            (datetime.datetime(2013, 4, 2), datetime.datetime(2013, 4, 3)),
                            (datetime.datetime(2013, 4, 8), datetime.datetime(2013, 4, 9)),  
                            (datetime.datetime(2013, 4, 26), datetime.datetime(2013, 4, 27)),
                            (datetime.datetime(2013, 4, 28, 12), datetime.datetime(2013, 4, 29, 12)),
                            ],
                        2 : [
                            (datetime.datetime(2013, 5, 21), datetime.datetime(2013, 5, 24)),
                            (datetime.datetime(2013, 6, 16), datetime.datetime(2013, 6, 19)),
                            (datetime.datetime(2013, 6, 20), datetime.datetime(2013, 6, 23)),
                            (datetime.datetime(2013, 6, 24), datetime.datetime(2013, 6, 28)),
                            ]
                    }
                },
            },
            "test" : {
                "ukdale" : {
                    "location" : "../../../datasets/ukdale/ukdale.h5",
                    "houses" : {
                        #5 : [(datetime.datetime(2014, 8, 1),datetime.datetime(2014, 8, 2))]
                        5 : [
                            (datetime.datetime(2014, 6, 30), datetime.datetime(2014, 7, 1)),
                            (datetime.datetime(2014, 7, 3), datetime.datetime(2014, 7, 4)),
                            (datetime.datetime(2014, 7, 6), datetime.datetime(2014, 7, 7)),
                            (datetime.datetime(2014, 7, 8), datetime.datetime(2014, 7, 9)),
                            (datetime.datetime(2014, 7, 11), datetime.datetime(2014, 7, 12)),
                            (datetime.datetime(2014, 7, 13), datetime.datetime(2014, 7, 14)),
                            (datetime.datetime(2014, 7, 19), datetime.datetime(2014, 7, 20)),
                            (datetime.datetime(2014, 7, 22), datetime.datetime(2014, 7, 23)),
                            (datetime.datetime(2014, 7, 27), datetime.datetime(2014, 7, 28)),
                            (datetime.datetime(2014, 7, 29), datetime.datetime(2014, 7, 30)),
                            (datetime.datetime(2014, 8, 1, 12), datetime.datetime(2014, 8, 2, 12)),
                            (datetime.datetime(2014, 8, 3), datetime.datetime(2014, 8, 4)),
                            ]
                    }
                }
            }
        }
    }

    run(experiment)

if __name__ == "__main__":
    run_experiment()
