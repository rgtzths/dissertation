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
        "fridge" : {
            "mains_columns" : [('power', 'apparent')],
            "appliance_columns" : [('power', 'apparent')],
            "methods" : {
                "LSTM" : LSTM_RNN({
                    'verbose' : 2,
                    "training_results_path" : "base_path/history/",
                    "results_path" : "base_path/results/LSTM/results_fridge.txt",
                    "checkpoint_file" : "base_path/models/LSTM/model_checkpoint_fridge.h5",
                    #"load_model_folder" : "base_path/models/LSTM/",
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
                    "training_results_path" : "base_path/history/",
                    "results_path" : "base_path/results/GRU/results_fridge.txt",
                    "checkpoint_file" : "base_path/models/GRU/model_checkpoint_fridge.h5",
                    #"load_model_folder" : "base_path/models/GRU/",
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
                "GRU2" : GRU2({
                    'verbose' : 2,
                    "training_results_path" : "base_path/history/",
                    "results_path" : "base_path/results/GRU2/results_fridge.txt",
                    "checkpoint_file" : "base_path/models/GRU2/model_checkpoint_fridge.h5",
                    #"load_model_folder" : "base_path/models/GRU2/",
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
                    "training_results_path" : "base_path/history/",
                    "results_path" : "base_path/results/ResNet/results_fridge.txt",
                    "checkpoint_file" : "base_path/models/ResNet/model_checkpoint_fridge.h5",
                    #"load_model_folder" : "base_path/models/ResNet/",
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
                "Seq2Point" : Seq2Point({'n_epochs':epochs,'batch_size':1024, "training_results_path" : "base_path/history/"})
            },
            "model_path" : "base_path/models/",
            "timestep" : 2,
            "train" : {
                "ukdale" : {
                    "location" : "../../../datasets/ukdale_classification/",
                    "houses" : {
                        "house_1" : [(datetime.datetime(2013, 5, 22), datetime.datetime(2013, 5, 23))], #"end" : datetime.datetime(2013, 7, 1)
                        "house_2" : [(datetime.datetime(2013, 5, 23), datetime.datetime(2013, 5, 24))] #"end" : datetime.datetime(2013, 7, 1)
                    }
                },
            },
            "test" : {
                "ukdale" : {
                    "location" : "../../../datasets/ukdale_classification/",
                    "houses" : {
                        "house_5" : [(datetime.datetime(2014, 6, 30),datetime.datetime(2014, 7, 1))]#"end" : datetime.datetime(2014, 7, 9)
                    }
                }
            }
        },
        "microwave" : {
            "mains_columns" : [('power', 'apparent')],
            "appliance_columns" : [('power', 'apparent')],
            "methods" : {
                "LSTM" : LSTM_RNN({
                    'verbose' : 2,
                    "training_results_path" : "base_path/history/",
                    "results_path" : "base_path/results/LSTM/results_microwave.txt",
                    "checkpoint_file" : "base_path/models/LSTM/model_checkpoint_microwave.h5",
                    #"load_model_folder" : "base_path/models/LSTM/",
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
                    "training_results_path" : "base_path/history/",
                    "results_path" : "base_path/results/GRU/results_microwave.txt",
                    "checkpoint_file" : "base_path/models/GRU/model_checkpoint_microwave.h5",
                    #"load_model_folder" : "base_path/models/GRU/",
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
                "GRU2" : GRU2({
                    'verbose' : 2,
                    "training_results_path" : "base_path/history/",
                    "results_path" : "base_path/results/GRU2/results_microwave.txt",
                    "checkpoint_file" : "base_path/models/GRU2/model_checkpoint_microwave.h5",
                    #"load_model_folder" : "base_path/models/GRU2/",
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
                    "training_results_path" : "base_path/history/",
                    "results_path" : "base_path/results/ResNet/results_microwave.txt",
                    "checkpoint_file" : "base_path/models/ResNet/model_checkpoint_microwave.h5",
                    #"load_model_folder" : "base_path/models/ResNet/",
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
                "Seq2Point" : Seq2Point({'n_epochs':epochs,'batch_size':1024, "training_results_path" : "base_path/history/"})
            },
            "model_path" : "base_path/models/",
            "timestep" : 2,
            "train" : {
                "ukdale" : {
                    "location" : "../../../datasets/ukdale_classification/",
                    "houses" : {
                        "house_1" : [(datetime.datetime(2013, 5, 23), datetime.datetime(2013, 5, 24))], #"end" : datetime.datetime(2013, 7, 1)
                        "house_2" : [(datetime.datetime(2013, 5, 23), datetime.datetime(2013, 5, 24))] #"end" : datetime.datetime(2013, 7, 1)
                    }
                },
            },
            "test" : {
                "ukdale" : {
                    "location" : "../../../datasets/ukdale_classification/",
                    "houses" : {
                        "house_5" : [(datetime.datetime(2014, 7, 1), datetime.datetime(2014, 7, 9))] #"end" : datetime.datetime(2013, 7, 9)
                    }
                }
            }
        },
        "dish_washer" : {
            "mains_columns" : [('power', 'apparent')],
            "appliance_columns" : [('power', 'apparent')],
            "methods" : {
                "LSTM" : LSTM_RNN({
                    'verbose' : 2,
                    "training_results_path" : "base_path/history/",
                    "results_path" : "base_path/results/LSTM/results_dish_washer.txt",
                    "checkpoint_file" : "base_path/models/LSTM/model_checkpoint_dish_washer.h5",
                    #"load_model_folder" : "base_path/models/LSTM/",
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
                    "training_results_path" : "base_path/history/",
                    "results_path" : "base_path/results/GRU/results_dish_washer.txt",
                    "checkpoint_file" : "base_path/models/GRU/model_checkpoint_dish_washer.h5",
                    #"load_model_folder" : "base_path/models/GRU/",
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
                "GRU2" : GRU2({
                    'verbose' : 2,
                    "training_results_path" : "base_path/history/",
                    "results_path" : "base_path/results/GRU2/results_dish_washer.txt",
                    "checkpoint_file" : "base_path/models/GRU2/model_checkpoint_dish_washer.h5",
                    #"load_model_folder" : "base_path/models/GRU2/",
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
                    "training_results_path" : "base_path/history/",
                    "results_path" : "base_path/results/ResNet/results_dish_washer.txt",
                    "checkpoint_file" : "base_path/models/ResNet/model_checkpoint_dish_washer.h5",
                    #"load_model_folder" : "base_path/models/ResNet/",
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
                "Seq2Point" : Seq2Point({'n_epochs':epochs,'batch_size':1024, "training_results_path" : "base_path/history/"})
            },
            "model_path" : "base_path/models/",
            "timestep" : 2,
            "train" : {
                "ukdale" : {
                    "location" : "../../../datasets/ukdale_classification/",
                    "houses" : {
                        "house_1" : [(datetime.datetime(2013, 5, 23), datetime.datetime(2013, 5, 24))], #"end" : datetime.datetime(2013, 7, 1)
                        "house_2" : [(datetime.datetime(2013, 5, 23), datetime.datetime(2013, 5, 24))] #"end" : datetime.datetime(2013, 7, 1)
                    }
                },
            },
            "test" : {
                "ukdale" : {
                    "location" : "../../../datasets/ukdale_classification/",
                    "houses" : {
                        "house_5" : [(datetime.datetime(2014, 6, 30), datetime.datetime(2014, 7, 9))] #"end" : datetime.datetime(2013, 7, 9)
                    }
                }
            }
        }
    }

    run(experiment)

if __name__ == "__main__":
    run_experiment()
