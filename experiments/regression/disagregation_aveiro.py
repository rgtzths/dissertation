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
from WindowGRU import WindowGRU

from  utils import create_path

aveiro_dataset = '../../../datasets/aveiro/avEiro.h5'

def run_heat_pump(base_path, timestep, epochs, batch_size, sequence_length):
    #Experiment Definition
    heatpump = {
        'power': {'mains': ['apparent'], 'appliance': ['apparent']},
        'sample_rate': timestep,
        'appliances': ['heat pump'],
        'methods': {
            'DAE':DAE({
                'n_epochs':epochs,
                'batch_size' : batch_size,
                'sequence_length': sequence_length,
                "training_history_folder" : base_path + "history/DAE/",
                "plots_folder" : base_path + "plots/DAE/",
                "file_prefix" : base_path + "temp_weights/DAE/",
                "on_threshold" : 50,
                "appliances" : {
                    "heat pump" : {
                    }
                },
            }),
            'WindowGRU':WindowGRU({
                'n_epochs':epochs,
                'batch_size' : batch_size,
                'sequence_length': sequence_length,
                "training_history_folder" : base_path + "history/WindowGRU/",
                "plots_folder" : base_path + "plots/WindowGRU/",
                "file_prefix" : base_path + "temp_weights/WindowGRU/",
                "on_threshold" : 50,
                "appliances" : {
                    "heat pump" : {
                    }
                },
            }),
            'Seq2Point':Seq2Point({
                'n_epochs':epochs,
                'batch_size' : batch_size,
                'sequence_length': sequence_length,
                "training_history_folder" : base_path + "history/Seq2Point/",
                "plots_folder" : base_path + "plots/Seq2Point/",
                "file_prefix" : base_path + "temp_weights/Seq2Point/",
                "on_threshold" : 50,
                "appliances" : {
                    "heat pump" : {
                    }
                },
            }),
            'Seq2Seq':Seq2Seq({
                'n_epochs':epochs,
                'sequence_length': sequence_length,
                'batch_size' : batch_size,
                "training_history_folder" : base_path + "history/Seq2Seq/",
                "plots_folder" : base_path + "plots/Seq2Seq/",
                "file_prefix" : base_path + "temp_weights/Seq2Seq/",
                "on_threshold" : 50,
                "appliances" : {
                    "heat pump" : {
                    }
                },
            }),
            "ResNet" : ResNet( {
                "verbose" : 2,
                "training_history_folder" : base_path + "history/ResNet/",
                "results_folder" : base_path + "results/ResNet/",
                "checkpoint_folder" : base_path + "temp_weights/ResNet/",
                "plots_folder" : base_path + "plots/ResNet/",
                "appliances" : {
                    "heat pump" : {
                        'timewindow' : timestep*sequence_length,
                        'timestep' : timestep,
                        'overlap' :  timestep*sequence_length - timestep,
                        'epochs' : epochs,
                        'batch_size' : batch_size,
                        'n_nodes' : 32,
                        'on_treshold' : 50,
                    }
                },
            }),
            "DeepGRU" : DeepGRU({
                'verbose' : 2,
                "training_history_folder" : base_path + "history/DeepGRU/",
                "results_folder" : base_path + "results/DeepGRU/",
                "checkpoint_folder" : base_path + "temp_weights/DeepGRU/",
                "plots_folder" : base_path + "plots/DeepGRU/",
                "appliances" : {
                    "heat pump" : {
                        'timewindow' : timestep*sequence_length,
                        'timestep' : timestep,
                        'overlap' :  timestep*sequence_length - timestep,
                        'epochs' : epochs,
                        'batch_size' : batch_size,
                        'n_nodes' : 90,
                        'on_treshold' : 50,
                    }
                },
            }),
            "MLP" : MLP( {
                "verbose" : 2,
                "training_history_folder" : base_path + "history/MLP/",
                "results_folder" : base_path + "results/MLP/",
                "checkpoint_folder" : base_path + "temp_weights/MLP/",
                "plots_folder" : base_path + "plots/MLP/",
                "appliances" : {
                    "heat pump" : {
                        'timewindow' : timestep*sequence_length,
                        'timestep' : timestep,
                        'overlap' :  timestep*sequence_length - timestep,
                        'epochs' : epochs,
                        'batch_size' : batch_size,
                        'feature_extractor' : "wt",
                        'on_treshold' : 50,
                    }
                },
            }),
            "MLP_Raw" : MLP( {
                "verbose" : 2,
                "training_history_folder" : base_path + "history/MLP_Raw/",
                "results_folder" : base_path + "results/MLP_Raw/",
                "checkpoint_folder" : base_path + "temp_weights/MLP_Raw/",
                "plots_folder" : base_path + "plots/MLP_Raw/",
                "appliances" : {
                    "heat pump" : {
                        'timewindow' : timestep*sequence_length,
                        'timestep' : timestep,
                        'overlap' :  timestep*sequence_length - timestep,
                        'epochs' : epochs,
                        'batch_size' : batch_size,
                        'on_treshold' : 50,
                    }     
                },
            }),
        },
        'train': {    
            'datasets': {
                'avEiro_train': {
                    "path": aveiro_dataset,
                    'buildings': {
                        1: {
                            'start_time': '2020-10-01',
                            'end_time': '2020-10-02'
                        }
                    }                
                },
                'avEiro_cv': {
                    "path": aveiro_dataset,
                    'buildings': {
                        1: {
                            'start_time': '2020-10-02',
                            'end_time': '2020-10-03'
                        }
                    }                
                },
            }
        },
        'test': {
            'datasets': {
                'avEiro': {
                    "path": aveiro_dataset,
                    'buildings': {
                        1: {
                            'start_time': '2021-01-04',
                            'end_time': '2021-01-05'
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

    for m in results.methods:
        create_path(base_path + "models/" + m)
        results.methods[m].save_model(base_path + "models/" + m)

def run_charger(base_path, timestep, epochs, batch_size, sequence_length):
    charger = {
        'power': {'mains': ['apparent'], 'appliance': ['apparent']},
        'sample_rate': timestep,
        'appliances': ['charger'],
        'methods': {
            'DAE':DAE({
                'n_epochs':epochs,
                'batch_size' : batch_size,
                'sequence_length': sequence_length,
                "training_history_folder" : base_path + "history/DAE/",
                "plots_folder" : base_path + "plots/DAE/",
                "file_prefix" : base_path + "temp_weights/DAE/",
                "on_threshold" : 50,
                "appliances" : {
                    "charger" : {
                    }
                },
            }),
            'WindowGRU':WindowGRU({
                'n_epochs':epochs,
                'batch_size' : batch_size,
                'sequence_length': sequence_length,
                "training_history_folder" : base_path + "history/WindowGRU/",
                "plots_folder" : base_path + "plots/WindowGRU/",
                "file_prefix" : base_path + "temp_weights/WindowGRU/",
                "on_threshold" : 50,
                "appliances" : {
                    "charger" : {
                    }
                },
            }),
            'Seq2Point':Seq2Point({
                'n_epochs':epochs,
                'batch_size' : batch_size,
                'sequence_length': sequence_length,
                "training_history_folder" : base_path + "history/Seq2Point/",
                "plots_folder" : base_path + "plots/Seq2Point/",
                "file_prefix" : base_path + "temp_weights/Seq2Point/",
                "on_threshold" : 50,
                "appliances" : {
                    "charger" : {
                    }
                },
            }),
            'Seq2Seq':Seq2Seq({
                'n_epochs':epochs,
                'sequence_length': sequence_length,
                'batch_size' : batch_size,
                "training_history_folder" : base_path + "history/Seq2Seq/",
                "plots_folder" : base_path + "plots/Seq2Seq/",
                "file_prefix" : base_path + "temp_weights/Seq2Seq/",
                "on_threshold" : 50,
                "appliances" : {
                    "charger" : {
                    }
                },
            }),
            "ResNet" : ResNet( {
                "verbose" : 2,
                "training_history_folder" : base_path + "history/ResNet/",
                "results_folder" : base_path + "results/ResNet/",
                "checkpoint_folder" : base_path + "temp_weights/ResNet/",
                "plots_folder" : base_path + "plots/ResNet/",
                "appliances" : {
                    "charger" : {
                        'timewindow' : timestep*sequence_length,
                        'timestep' : timestep,
                        'overlap' :  timestep*sequence_length - timestep,
                        'epochs' : epochs,
                        'batch_size' : batch_size,
                        'n_nodes' : 32,
                        'on_treshold' : 50,
                    }
                },
            }),
            "DeepGRU" : DeepGRU({
                'verbose' : 2,
                "training_history_folder" : base_path + "history/DeepGRU/",
                "results_folder" : base_path + "results/DeepGRU/",
                "checkpoint_folder" : base_path + "temp_weights/DeepGRU/",
                "plots_folder" : base_path + "plots/DeepGRU/",
                "appliances" : {
                    "charger" : {
                        'timewindow' : timestep*sequence_length,
                        'timestep' : timestep,
                        'overlap' :  timestep*sequence_length - timestep,
                        'epochs' : epochs,
                        'batch_size' : batch_size,
                        'n_nodes' : 90,
                        'on_treshold' : 50,
                    }
                },
            }),
            "MLP" : MLP( {
                "verbose" : 2,
                "training_history_folder" : base_path + "history/MLP/",
                "results_folder" : base_path + "results/MLP/",
                "checkpoint_folder" : base_path + "temp_weights/MLP/",
                "plots_folder" : base_path + "plots/MLP/",
                "appliances" : {
                    "charger" : {
                        'timewindow' : timestep*sequence_length,
                        'timestep' : timestep,
                        'overlap' :  timestep*sequence_length - timestep,
                        'epochs' : epochs,
                        'batch_size' : batch_size,
                        'feature_extractor' : "wt",
                        'on_treshold' : 50,
                    }
                },
            }),
            "MLP_Raw" : MLP( {
                "verbose" : 2,
                "training_history_folder" : base_path + "history/MLP_Raw/",
                "results_folder" : base_path + "results/MLP_Raw/",
                "checkpoint_folder" : base_path + "temp_weights/MLP_Raw/",
                "plots_folder" : base_path + "plots/MLP_Raw/",
                "appliances" : {
                    "charger" : {
                        'timewindow' : timestep*sequence_length,
                        'timestep' : timestep,
                        'overlap' :  timestep*sequence_length - timestep,
                        'epochs' : epochs,
                        'batch_size' : batch_size,
                        'on_treshold' : 50,
                    }     
                },
            }),
        },
        'train': {    
            'datasets': {
                'avEiro_train': {
                    "path": aveiro_dataset,
                    'buildings': {
                        1: {
                            'start_time': '2020-11-14',
                            'end_time': '2020-11-15'
                        }
                    }                
                },
                'avEiro_cv': {
                    "path": aveiro_dataset,
                    'buildings': {
                        1: {
                            'start_time': '2020-11-15',
                            'end_time': '2020-11-16'
                        }
                    }                
                },
            }
        },
        'test': {
            'datasets': {
                'avEiro': {
                    "path": aveiro_dataset,
                    'buildings': {
                        1: {
                            'start_time': '2021-01-04',
                            'end_time': '2021-01-05'
                        }
                    }
                }
            },
            'metrics':['mae', 'rmse']
        }
    }

    ## Training and testing Car charger ####

    results = API(charger)

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

    for m in results.methods:
        create_path(base_path + "models/" + m)
        results.methods[m].save_model(base_path + "models/" + m)


if __name__ == "__main__":
    base_path= "/home/rteixeira/temp/"
    #base_path = "/home/user/article_results/"
    #base_path = "/home/atnoguser/transfer_results/ukdale/"
    epochs = 1
    batch_size = 256
    sequence_length = 299
    timestep = 2

    run_heat_pump(base_path, timestep, epochs, batch_size, sequence_length)
    run_charger(base_path, timestep, epochs, batch_size, sequence_length)