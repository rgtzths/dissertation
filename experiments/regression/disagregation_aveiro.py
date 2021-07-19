from nilmtk.api import API

import sys
sys.path.insert(1, "../../nilmtk-contrib")
sys.path.insert(1, "../../regression_models")
sys.path.insert(1, "../../utils")
sys.path.insert(1, "../../feature_extractors")

from dae import DAE
from seq2point import Seq2Point
from seq2seq import Seq2Seq

from resnet import ResNet
from deep_gru import DeepGRU
from mlp_dwt import MLP

base_path= "/home/rteixeira/thesis_results/regression/"
#base_path = "/home/user/thesis_results/regression/"
epochs = 1
timestep = 2
timewindow = 90 * timestep
overlap = timewindow - timestep

#Experiment Definition
heatpump = {
    'power': {'mains': ['apparent'], 'appliance': ['apparent']},
    'sample_rate': timestep,
    'appliances': ['heat pump'],
    'methods': {
        'DAE':DAE({'n_epochs':epochs,'batch_size':1024}),
        'Seq2Point':Seq2Point({'n_epochs':epochs,'batch_size':1024}),
        'Seq2Seq':Seq2Seq({'n_epochs':epochs,'batch_size':1024}),
        "ResNet" : ResNet( {
            "verbose" : 2,
            "training_history_folder" : base_path + "history/ResNet/",
            "results_folder" : base_path + "results/ResNet/",
            "checkpoint_folder" : base_path + "models/ResNet/",
            "plots_folder" : base_path + "plots/ResNet/",
            #"load_model_folder" : base_path + "models/ResNet/",
            "appliances" : {
                "heat pump" : {
                    'timewindow' : timewindow,
                    'timestep' : timestep,
                    'overlap' : overlap,
                    'epochs' : epochs,
                    'batch_size' : 1024,
                    'n_nodes' : 90
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
                    'timewindow' : timewindow,
                    'timestep' : timestep,
                    'overlap' : overlap,
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
                    'timewindow' : timewindow,
                    'timestep' : timestep,
                    'overlap' : overlap,
                    'epochs' : epochs,
                    'batch_size' : 1024,
                    'feature_extractor' : "dwt"
                }
            },
            "predicted_column": ("power", "apparent"), 
        }),
    },
    'train': {    
        'datasets': {
            'avEiro': {
                'path': '../../../datasets/avEiro_h5/avEiro.h5',
                'buildings': {
                    1: {
                        'start_time': '2020-10-05',
                        'end_time': '2020-10-09'
                    }
                }                
            },
            'avEiro2': {
                'path': '../../../datasets/avEiro_h5/avEiro.h5',
                'buildings': {
                    1: {
                        'start_time': '2020-10-20',
                        'end_time': '2020-10-27'
                    }
                }                
            },
            'avEiro3': {
                'path': '../../../datasets/avEiro_h5/avEiro.h5',
                'buildings': {
                    1: {
                        'start_time': '2020-11-01',
                        'end_time': '2020-11-08'
                    }
                }                
            },
            'avEiro4': {
                'path': '../../../datasets/avEiro_h5/avEiro.h5',
                'buildings': {
                    1: {
                        'start_time': '2020-12-09',
                        'end_time': '2020-12-17'
                    }
                }                
            },
        }
    },
    'test': {
        'datasets': {
            'avEiro': {
                'path': '../../../datasets/avEiro_h5/avEiro.h5',
                'buildings': {
                    1: {
                        'start_time': '2021-01-15',
                        'end_time': '2021-02-05'
                    }
                }
            }
        },
        'metrics':['mae', 'rmse']
    }
}

charger = {
    'power': {'mains': ['apparent'], 'appliance': ['apparent']},
    'sample_rate': timestep,
    'appliances': ['charger'],
    'methods': {
        'DAE':DAE({'n_epochs':epochs,'batch_size':1024}),
        'Seq2Point':Seq2Point({'n_epochs':epochs,'batch_size':1024}),
        'Seq2Seq':Seq2Seq({'n_epochs':epochs,'batch_size':1024}),
        "ResNet" : ResNet( {
            "verbose" : 2,
            "training_history_folder" : base_path + "history/ResNet/",
            "results_folder" : base_path + "results/ResNet/",
            "checkpoint_folder" : base_path + "models/ResNet/",
            "plots_folder" : base_path + "plots/ResNet/",
            #"load_model_folder" : base_path + "models/ResNet/",
            "appliances" : {
                "charger" : {
                    'timewindow' : timewindow,
                    'timestep' : timestep,
                    'overlap' : overlap,
                    'epochs' : epochs,
                    'batch_size' : 1024,
                    'n_nodes' : 90
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
                    'timewindow' : timewindow,
                    'timestep' : timestep,
                    'overlap' : overlap,
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
                    'timewindow' : timewindow,
                    'timestep' : timestep,
                    'overlap' : overlap,
                    'epochs' : epochs,
                    'batch_size' : 1024,
                    'feature_extractor' : "dwt"
                }
            },
            "predicted_column": ("power", "apparent"), 
        }),
    },
    'train': {    
        'datasets': {
            'avEiro': {
                'path': '../../../datasets/avEiro_h5/avEiro.h5',
                'buildings': {
                    1: {
                        'start_time': '2020-11-14',
                        'end_time': '2020-11-15'
                    }
                }                
            },
            'avEiro2': {
                'path': '../../../datasets/avEiro_h5/avEiro.h5',
                'buildings': {
                    1: {
                        'start_time': '2020-11-16',
                        'end_time': '2020-11-20'
                    }
                }                
            },
            'avEiro3': {
                'path': '../../../datasets/avEiro_h5/avEiro.h5',
                'buildings': {
                    1: {
                        'start_time': '2020-11-23',
                        'end_time': '2020-11-29'
                    }
                }                
            },
            'avEiro4': {
                'path': '../../../datasets/avEiro_h5/avEiro.h5',
                'buildings': {
                    1: {
                        'start_time': '2020-12-02',
                        'end_time': '2020-12-06'
                    }
                }                
            },
            'avEiro4': {
                'path': '../../../datasets/avEiro_h5/avEiro.h5',
                'buildings': {
                    1: {
                        'start_time': '2020-12-09',
                        'end_time': '2020-12-13'
                    }
                }                
            },
            'avEiro5': {
                'path': '../../../datasets/avEiro_h5/avEiro.h5',
                'buildings': {
                    1: {
                        'start_time': '2020-12-16',
                        'end_time': '2020-12-21'
                    }
                }                
            },
            'avEiro6': {
                'path': '../../../datasets/avEiro_h5/avEiro.h5',
                'buildings': {
                    1: {
                        'start_time': '2021-01-02',
                        'end_time': '2020-01-04'
                    }
                }                
            },
            'avEiro7': {
                'path': '../../../datasets/avEiro_h5/avEiro.h5',
                'buildings': {
                    1: {
                        'start_time': '2021-01-06',
                        'end_time': '2020-01-06'
                    }
                }                
            },
        }
    },
    'test': {
        'datasets': {
            'avEiro': {
                'path': '../../../datasets/avEiro_h5/avEiro.h5',
                'buildings': {
                    1: {
                        'start_time': '2021-01-06',
                        'end_time': '2021-01-21'
                    }
                }
            }
        },
        'metrics':['mae', 'rmse']
    }
}

## Training and testing Heat Pump ####
results = API(heatpump)

#Get all the results in the experiment and print them.
errors_keys = results.errors_keys
errors = results.errors

for app in results.appliances:
    f = open(base_path + "results_" + app.replace(" ", "_") + ".txt", "w")
    for classifier in errors[0].columns:
        f.write(5*"-" + classifier + "-"*5 + "\n")
        for i in range(len(errors)):
            f.write(errors_keys[i].split("_")[-1].upper() + " : " + str(errors[i][classifier][app]) + "\n")

##################################################

## Training and testing Car charger ####

results = API(heatpump)

#Get all the results in the experiment and print them.
errors_keys = results.errors_keys
errors = results.errors

for app in results.appliances:
    f = open(base_path + "results_" + app.replace(" ", "_") + ".txt", "w")
    for classifier in errors[0].columns:
        f.write(5*"-" + classifier + "-"*5 + "\n")
        for i in range(len(errors)):
            f.write(errors_keys[i].split("_")[-1].upper() + " : " + str(errors[i][classifier][app]) + "\n")

##################################################