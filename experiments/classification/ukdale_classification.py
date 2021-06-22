import datetime
import warnings
warnings.filterwarnings("ignore")

from scipy.stats import randint, loguniform

from run_experiment import run

import sys
sys.path.insert(1, "../../classification_models")

from seq2point import Seq2Point
from gru import GRU_RNN
from deep_gru import DeepGRU
from lstm import LSTM_RNN
from resnet import ResNet
from mlp_dwt import MLP

#base_path= "/home/rteixeira/thesis_results/"
base_path = "/home/user/thesis_results/"
epochs = 1

def run_experiment():

    #Experiment Definition
    experiment = {
        "fridge" : {
            "mains_columns" : [('power', 'apparent')],
            "appliance_columns" : [('power', 'apparent')],
            "methods" : {
                #"LSTM" : LSTM_RNN({
                #    'verbose' : 2,
                #    "training_results_path" : base_path+"history/",
                #    "results_path" : base_path+"results/LSTM/results_fridge.txt",
                #    "checkpoint_file" : base_path+"models/LSTM/model_checkpoint_fridge.h5",
                #    #"load_model_folder" : base_path+"models/LSTM/",
                #    "appliances" : {
                #        "fridge" : {
                #            'timewindow' : 180,
                #            'timestep' : 2,
                #            'overlap' : 178,
                #            'epochs' : epochs,
                #            'batch_size' : 1024,
                #            'n_nodes' : 90,
                #        }
                #    },
                #}),
                #"GRU" : GRU_RNN({
                #    'verbose' : 2,
                #    "training_results_path" : base_path+"history/",
                #    "results_path" : base_path+"results/GRU/results_fridge.txt",
                #    "checkpoint_file" : base_path+"models/GRU/model_checkpoint_fridge.h5",
                #    #"load_model_folder" : base_path+"models/GRU/",
                #    "appliances" : {
                #        "fridge" : {
                #            'timewindow' : 180,
                #            'timestep' : 2,
                #            'overlap' : 178,
                #            'epochs' : epochs,
                #            'batch_size' : 1024,
                #            'n_nodes' : 90,
                #        }
                #    },
                #}),
                "DeepGRU" : DeepGRU({
                    'verbose' : 2,
                    "training_results_path" : base_path+"history/",
                    "results_path" : base_path+"results/DeepGRU/results_fridge.txt",
                    "checkpoint_file" : base_path+"models/DeepGRU/model_checkpoint_fridge.h5",
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
                    "training_results_path" : base_path+"history/",
                    "results_path" : base_path+"results/ResNet/results_fridge.txt",
                    "checkpoint_file" : base_path+"models/ResNet/model_checkpoint_fridge.h5",
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
                #"MLP" : MLP( {
                #    "verbose" : 2,
                #    "training_results_path" : base_path + "history/",
                #    "results_path" : base_path + "results/MLP/results_fridge.txt",
                #    "checkpoint_file" : base_path + "models/MLP/model_checkpoint_fridge.h5",
                #    #"load_model_folder" : base_path + "models/MLP/",
                #    "appliances" : {
                #        "fridge" : {
                #            'timewindow' : 180,
                #            'timestep' : 2,
                #            'overlap' : 178,
                #            'epochs' : epochs,
                #            'batch_size' : 1024,
                #            'feature_extractor' : "dwt"
                #        }
                #    },
                #    "predicted_column": ("power", "apparent"), 
                #}),
            },
            "model_path" : base_path+"models/",
            "timestep" : 2,
            "train" : {
                "ukdale" : {
                    "location" : "../../../datasets/ukdale_classification/",
                    "houses" : {
                        #"house_1" : [(datetime.datetime(2014, 2, 22), datetime.datetime(2014, 2, 23))],
                        #"house_2" : [(datetime.datetime(2013, 5, 23), datetime.datetime(2013, 5, 24))]
                        "house_1" : [
                            (datetime.datetime(2014, 2, 20), datetime.datetime(2014, 2, 28)),
                            (datetime.datetime(2016, 9, 10), datetime.datetime(2016, 9, 17))
                            ],
                        "house_2" : [
                            (datetime.datetime(2013, 5, 23), datetime.datetime(2013, 5, 30)),
                            (datetime.datetime(2013, 7, 10), datetime.datetime(2013, 7, 17))
                            ]
                    }
                },
            },
            "test" : {
                "ukdale" : {
                    "location" : "../../../datasets/ukdale_classification/",
                    "houses" : {
                        #"house_5" : [(datetime.datetime(2014, 8, 1),datetime.datetime(2014, 8, 2))]
                        "house_5" : [
                            (datetime.datetime(2014, 8, 1),datetime.datetime(2014, 8, 8)),
                            (datetime.datetime(2014, 9, 8),datetime.datetime(2014, 9, 15))
                            ]
                    }
                }
            }
        },
        "microwave" : {
            "mains_columns" : [('power', 'apparent')],
            "appliance_columns" : [('power', 'apparent')],
            "methods" : {
                #"LSTM" : LSTM_RNN({
                #    'verbose' : 2,
                #    "training_results_path" : base_path+"history/",
                #    "results_path" : base_path+"results/LSTM/results_microwave.txt",
                #    "checkpoint_file" : base_path+"models/LSTM/model_checkpoint_microwave.h5",
                #    #"load_model_folder" : base_path+"models/LSTM/",
                #    "appliances" : {
                #        "microwave" : {
                #            'timewindow' : 180,
                #            'timestep' : 2,
                #            'overlap' : 178,
                #            'epochs' : epochs,
                #            'batch_size' : 1024,
                #            'n_nodes' : 90,
                #        }
                #    },
                #}),
                #"GRU" : GRU_RNN({
                #    'verbose' : 2,
                #    "training_results_path" : base_path+"history/",
                #    "results_path" : base_path+"results/GRU/results_microwave.txt",
                #    "checkpoint_file" : base_path+"models/GRU/model_checkpoint_microwave.h5",
                #    #"load_model_folder" : base_path+"models/GRU/",
                #    "appliances" : {
                #        "microwave" : {
                #            'timewindow' : 180,
                #            'timestep' : 2,
                #            'overlap' : 178,
                #            'epochs' : epochs,
                #            'batch_size' : 1024,
                #            'n_nodes' : 90,
                #        }
                #    },
                #}),
                "DeepGRU" : DeepGRU({
                    'verbose' : 2,
                    "training_results_path" : base_path+"history/",
                    "results_path" : base_path+"results/DeepGRU/results_microwave.txt",
                    "checkpoint_file" : base_path+"models/DeepGRU/model_checkpoint_microwave.h5",
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
                    "training_results_path" : base_path+"history/",
                    "results_path" : base_path+"results/ResNet/results_microwave.txt",
                    "checkpoint_file" : base_path+"models/ResNet/model_checkpoint_microwave.h5",
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
                #"MLP" : MLP( {
                #    "verbose" : 2,
                #    "training_results_path" : base_path + "history/",
                #    "results_path" : base_path + "results/MLP/results_microwave.txt",
                #    "checkpoint_file" : base_path + "models/MLP/model_checkpoint_microwave.h5",
                #    #"load_model_folder" : base_path + "models/MLP/",
                #    "appliances" : {
                #        "microwave" : {
                #            'timewindow' : 180,
                #            'timestep' : 2,
                #            'overlap' : 178,
                #            'epochs' : epochs,
                #            'batch_size' : 1024,
                #            'feature_extractor' : "dwt"
                #        }
                #    },
                #    "predicted_column": ("power", "apparent"), 
                #}),
            },
            "model_path" : base_path+"models/",
            "timestep" : 2,
            "train" : {
                "ukdale" : {
                    "location" : "../../../datasets/ukdale_classification/",
                    "houses" : {
                        #"house_1" : [(datetime.datetime(2014, 2, 22), datetime.datetime(2014, 2, 23))],
                        #"house_2" : [(datetime.datetime(2013, 5, 23), datetime.datetime(2013, 5, 24))]
                        "house_1" : [
                            (datetime.datetime(2014, 2, 20), datetime.datetime(2014, 2, 28)),
                            (datetime.datetime(2016, 9, 10), datetime.datetime(2016, 9, 17))
                            ],
                        "house_2" : [
                            (datetime.datetime(2013, 5, 23), datetime.datetime(2013, 5, 30)),
                            (datetime.datetime(2013, 7, 10), datetime.datetime(2013, 7, 17))
                            ]
                    }
                },
            },
            "test" : {
                "ukdale" : {
                    "location" : "../../../datasets/ukdale_classification/",
                    "houses" : {
                        #"house_5" : [(datetime.datetime(2014, 8, 1),datetime.datetime(2014, 8, 2))]
                        "house_5" : [
                            (datetime.datetime(2014, 8, 1),datetime.datetime(2014, 8, 8)),
                            (datetime.datetime(2014, 9, 8),datetime.datetime(2014, 9, 15))
                            ]
                    }
                }
            }
        },
        "dish_washer" : {
            "mains_columns" : [('power', 'apparent')],
            "appliance_columns" : [('power', 'apparent')],
            "methods" : {
                #"LSTM" : LSTM_RNN({
                #    'verbose' : 2,
                #    "training_results_path" : base_path+"history/",
                #    "results_path" : base_path+"results/LSTM/results_dish_washer.txt",
                #    "checkpoint_file" : base_path+"models/LSTM/model_checkpoint_dish_washer.h5",
                #    #"load_model_folder" : base_path+"models/LSTM/",
                #    "appliances" : {
                #        "dish_washer" : {
                #            'timewindow' : 180,
                #            'timestep' : 2,
                #            'overlap' : 178,
                #            'epochs' : epochs,
                #            'batch_size' : 1024,
                #            'n_nodes' : 90,
                #        }
                #    },
                #}),
                #"GRU" : GRU_RNN({
                #    'verbose' : 2,
                #    "training_results_path" : base_path+"history/",
                #    "results_path" : base_path+"results/GRU/results_dish_washer.txt",
                #    "checkpoint_file" : base_path+"models/GRU/model_checkpoint_dish_washer.h5",
                #    #"load_model_folder" : base_path+"models/GRU/",
                #    "appliances" : {
                #        "dish_washer" : {
                #            'timewindow' : 180,
                #            'timestep' : 2,
                #            'overlap' : 178,
                #            'epochs' : epochs,
                #            'batch_size' : 1024,
                #            'n_nodes' : 90,
                #        }
                #    },
                #}),
                "DeepGRU" : DeepGRU({
                    'verbose' : 2,
                    "training_results_path" : base_path+"history/",
                    "results_path" : base_path+"results/DeepGRU/results_dish_washer.txt",
                    "checkpoint_file" : base_path+"models/DeepGRU/model_checkpoint_dish_washer.h5",
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
                    "training_results_path" : base_path+"history/",
                    "results_path" : base_path+"results/ResNet/results_dish_washer.txt",
                    "checkpoint_file" : base_path+"models/ResNet/model_checkpoint_dish_washer.h5",
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
                #"MLP" : MLP( {
                #    "verbose" : 2,
                #    "training_results_path" : base_path + "history/",
                #    "results_path" : base_path + "results/MLP/results_dish_washer.txt",
                #    "checkpoint_file" : base_path + "models/MLP/model_checkpoint_dish_washer.h5",
                #    #"load_model_folder" : base_path + "models/MLP/",
                #    "appliances" : {
                #        "dish_washer" : {
                #            'timewindow' : 180,
                #            'timestep' : 2,
                #            'overlap' : 178,
                #            'epochs' : epochs,
                #            'batch_size' : 1024,
                #            'feature_extractor' : "dwt"
                #        }
                #    },
                #    "predicted_column": ("power", "apparent"), 
                #}),
            },
            "model_path" : base_path+"models/",
            "timestep" : 2,
            "train" : {
                "ukdale" : {
                    "location" : "../../../datasets/ukdale_classification/",
                    "houses" : {
                        #"house_1" : [(datetime.datetime(2014, 2, 22), datetime.datetime(2014, 2, 23))],
                        #"house_2" : [(datetime.datetime(2015, 9, 23), datetime.datetime(2015, 9, 24))]
                        "house_1" : [
                            (datetime.datetime(2014, 2, 20), datetime.datetime(2014, 2, 28)),
                            (datetime.datetime(2016, 9, 10), datetime.datetime(2016, 9, 17))
                            ],
                        "house_2" : [
                            (datetime.datetime(2015, 9, 23), datetime.datetime(2015, 9, 30)),
                            (datetime.datetime(2016, 1, 10), datetime.datetime(2016, 1, 17))
                            ]
                    }
                },
            },
            "test" : {
                "ukdale" : {
                    "location" : "../../../datasets/ukdale_classification/",
                    "houses" : {
                        #"house_5" : [(datetime.datetime(2014, 8, 1),datetime.datetime(2014, 8, 2))]
                        "house_5" : [
                            (datetime.datetime(2014, 8, 1),datetime.datetime(2014, 8, 8)),
                            (datetime.datetime(2014, 9, 8),datetime.datetime(2014, 9, 15))
                            ]
                    }
                }
            }
        }
    }

    run(experiment)

if __name__ == "__main__":
    run_experiment()
