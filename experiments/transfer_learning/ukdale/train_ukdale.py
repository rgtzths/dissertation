from nilmtk.api import API

import sys
sys.path.insert(1, "../../../nilmtk-contrib")
sys.path.insert(1, "../../../regression_models")
sys.path.insert(1, "../../../utils")
sys.path.insert(1, "../../../feature_extractors")

from utils import create_path
from dae import DAE
from seq2point import Seq2Point
from seq2seq import Seq2Seq
from WindowGRU import WindowGRU

from resnet import ResNet
from deep_gru import DeepGRU
from mlp_dwt import MLP

def run_fridge(base_path, timestep, epochs, batch_size, sequence_length):
    fridge = {
        'power': {'mains': ['active'],'appliance': ['active']},
        'sample_rate': timestep,
        'appliances': ['fridge'],
        'methods': {
            #'DAE':DAE({
            #    'n_epochs':epochs,
            #    'batch_size' : batch_size,
            #    'sequence_length': sequence_length,
            #    "training_history_folder" : base_path + "history/DAE/",
            #    "plots_folder" : base_path + "plots/DAE/",
            #    "file_prefix" : base_path + "temp_weights/DAE/",
            #    "on_threshold" : 200,
            #    "appliances" : {
            #        "fridge" : {
            #            'transfer_path': base_path + "models/DAE/fridge.h5"
            #        }
            #    },
            #}),
            #'WindowGRU':WindowGRU({
            #    'n_epochs':epochs,
            #    'batch_size' : batch_size,
            #    'sequence_length': sequence_length,
            #    "training_history_folder" : base_path + "history/WindowGRU/",
            #    "plots_folder" : base_path + "plots/WindowGRU/",
            #    "file_prefix" : base_path + "temp_weights/WindowGRU/",
            #    "on_threshold" : 200,
            #    "appliances" : {
            #        "fridge" : {
            #            'transfer_path': base_path + "models/WindowGRU/fridge.h5"
            #        }
            #    },
            #}),
            #'Seq2Point':Seq2Point({
            #    'n_epochs':epochs,
            #    'batch_size' : batch_size,
            #    'sequence_length': sequence_length,
            #    "training_history_folder" : base_path + "history/Seq2Point/",
            #    "plots_folder" : base_path + "plots/Seq2Point/",
            #    "file_prefix" : base_path + "temp_weights/Seq2Point/",
            #    "on_threshold" : 200,
            #    "appliances" : {
            #        "fridge" : {
            #            'transfer_path': base_path + "models/Seq2Point/fridge.h5"
            #        }
            #    },
            #}),
            #'Seq2Seq':Seq2Seq({
            #    'n_epochs':epochs,
            #    'sequence_length': sequence_length,
            #    'batch_size' : batch_size,
            #    "training_history_folder" : base_path + "history/Seq2Seq/",
            #    "plots_folder" : base_path + "plots/Seq2Seq/",
            #    "file_prefix" : base_path + "temp_weights/Seq2Seq/",
            #    "on_threshold" : 200,
            #    "appliances" : {
            #        "fridge" : {
            #            'transfer_path': base_path + "models/Seq2Seq/fridge.h5"
            #        }
            #    },
            #}),
            #"ResNet" : ResNet( {
            #    "verbose" : 2,
            #    "training_history_folder" : base_path + "history/ResNet/",
            #    "results_folder" : base_path + "results/ResNet/",
            #    "checkpoint_folder" : base_path + "temp_weights/ResNet/",
            #    "plots_folder" : base_path + "plots/ResNet/",
            #    #"load_model_folder" : base_path + "temp_weights/ResNet/",
            #    "appliances" : {
            #        "fridge" : {
            #            'timewindow' : timestep*sequence_length,
            #            'timestep' : timestep,
            #            'overlap' :  timestep*sequence_length - timestep,
            #            'epochs' : epochs,
            #            'batch_size' : batch_size,
            #            'n_nodes' : 32,
            #            'on_treshold' : 50,
            #            'transfer_path': base_path + "models/ResNet/fridge.h5"
            #        }
            #    },
            #}),
            #"DeepGRU" : DeepGRU({
            #    'verbose' : 2,
            #    "training_history_folder" : base_path + "history/DeepGRU/",
            #    "results_folder" : base_path + "results/DeepGRU/",
            #    "checkpoint_folder" : base_path + "temp_weights/DeepGRU/",
            #    "plots_folder" : base_path + "plots/DeepGRU/",
            #    #"load_model_folder" : base_path + "temp_weights/DeepGRU/",
            #    "appliances" : {
            #        "fridge" : {
            #            'timewindow' : timestep*sequence_length,
            #            'timestep' : timestep,
            #            'overlap' :  timestep*sequence_length - timestep,
            #            'epochs' : epochs,
            #            'batch_size' : batch_size,
            #            'n_nodes' : 90,
            #            'on_treshold' : 50,
            #            'transfer_path': base_path + "models/DeepGRU/fridge.h5"
            #        }
            #    },
            #}),
            "MLP" : MLP( {
                "verbose" : 2,
                "training_history_folder" : base_path + "history/MLP/",
                "results_folder" : base_path + "results/MLP/",
                "checkpoint_folder" : base_path + "temp_weights/MLP/",
                "plots_folder" : base_path + "plots/MLP/",
                #"load_model_folder" : base_path + "temp_weights/MLP/",
                "appliances" : {
                    "fridge" : {
                        'timewindow' : timestep*sequence_length,
                        'timestep' : timestep,
                        'overlap' :  timestep*sequence_length - timestep,
                        'epochs' : epochs,
                        'batch_size' : batch_size,
                        'feature_extractor' : "wt",
                        'on_treshold' : 50,
                        'transfer_path': base_path + "models/MLP/fridge.h5"
                    }
                },
            }),
            #"MLP_Raw" : MLP( {
            #    "verbose" : 2,
            #    "training_history_folder" : base_path + "history/MLP_Raw/",
            #    "results_folder" : base_path + "results/MLP_Raw/",
            #    "checkpoint_folder" : base_path + "temp_weights/MLP_Raw/",
            #    "plots_folder" : base_path + "plots/MLP_Raw/",
            #    #"load_model_folder" : base_path + "temp_weights/MLP_Raw/",
            #    "appliances" : {
            #        "fridge" : {
            #            'timewindow' : timestep*sequence_length,
            #            'timestep' : timestep,
            #            'overlap' :  timestep*sequence_length - timestep,
            #            'epochs' : epochs,
            #            'batch_size' : batch_size,
            #            'on_treshold' : 50,
            #            'transfer_path': base_path + "models/MLP_Raw/fridge.h5"
            #        }     
            #    },
            #}),
        },
        'train': {    
            'datasets': {
                'UKDale': {
                    'path': '../../../../../datasets/ukdale/ukdale.h5',
                    'buildings': {
                        1: {
                            'start_time': "2013-06-17",
                            'end_time': "2013-06-18",
                        },
                        2: {
                            'start_time': "2013-06-17",
                            'end_time': "2013-06-18",
                        }           
                    }
                },
            }
        },
        'test': {
            'datasets': {
                'UKDale': {
                    'path': '../../../../../datasets/ukdale/ukdale.h5',
                    'buildings': {
                        5: {
                            'start_time': "2014-08-01",
                            'end_time': "2014-08-02"
                        } 
                    }
                },
            },
            'metrics':['mae', 'rmse']
        }
    }

    ### Training and testing fridge ####
    results = API(fridge)

    #Get all the results in the experiment and print them.
    errors_keys = results.errors_keys
    errors = results.errors

    for app in results.appliances:
        f = open(base_path + "results_" + app.replace(" ", "_") + ".txt", "w")
        for classifier in errors[0].columns:
            f.write(5*"-" + classifier + "-"*5 + "\n")
            for i in range(len(errors)):
                f.write(errors_keys[i].split("_")[-1].upper() + " : " + str(errors[i][classifier][app]) + "\n")

    ###################################################
    for m in results.methods:
        create_path(base_path + "models/" + m)
        results.methods[m].save_model(base_path + "models/" + m)

def run_kettle(base_path, timestep, epochs, batch_size, sequence_length):
    kettle = {
        'power': {'mains': ['active'],'appliance': ['active']},
        'sample_rate': timestep,
        'appliances': ['kettle'],
        'methods': {
            'DAE':DAE({
                'n_epochs':epochs,
                'batch_size' : batch_size,
                'sequence_length': sequence_length,
                "training_history_folder" : base_path + "history/DAE/",
                "plots_folder" : base_path + "plots/DAE/",
                "file_prefix" : base_path + "temp_weights/DAE/",
                "on_threshold" : 2000
            }),
            'Seq2Point':Seq2Point({
                'n_epochs':epochs,
                'batch_size' : batch_size,
                'sequence_length': sequence_length,
                "training_history_folder" : base_path + "history/Seq2Point/",
                "plots_folder" : base_path + "plots/Seq2Point/",
                "file_prefix" : base_path + "temp_weights/Seq2Point/",
                "on_threshold" : 2000
            }),
            'Seq2Seq':Seq2Seq({
                'n_epochs':epochs,
                'sequence_length': sequence_length,
                'batch_size' : batch_size,
                "training_history_folder" : base_path + "history/Seq2Seq/",
                "plots_folder" : base_path + "plots/Seq2Seq/",
                "file_prefix" : base_path + "temp_weights/Seq2Seq/",
                "on_threshold" : 2000
            }),
            "ResNet" : ResNet( {
                "verbose" : 2,
                "training_history_folder" : base_path + "history/ResNet/",
                "results_folder" : base_path + "results/ResNet/",
                "checkpoint_folder" : base_path + "temp_weights/ResNet/",
                "plots_folder" : base_path + "plots/ResNet/",
                #"load_model_folder" : base_path + "temp_weights/ResNet/",
                "appliances" : {
                    "kettle" : {
                        'timewindow' : timestep*sequence_length,
                        'timestep' : timestep,
                        'overlap' :  timestep*sequence_length - timestep,
                        'epochs' : epochs,
                        'batch_size' : batch_size,
                        'n_nodes' : 32,
                        'on_treshold' : 2000
                    }
                },
            }),
            "DeepGRU" : DeepGRU({
                'verbose' : 2,
                "training_history_folder" : base_path + "history/DeepGRU/",
                "results_folder" : base_path + "results/DeepGRU/",
                "checkpoint_folder" : base_path + "temp_weights/DeepGRU/",
                "plots_folder" : base_path + "plots/DeepGRU/",
                #"load_model_folder" : base_path + "temp_weights/DeepGRU/",
                "appliances" : {
                    "kettle" : {
                        'timewindow' : timestep*sequence_length,
                        'timestep' : timestep,
                        'overlap' :  timestep*sequence_length - timestep,
                        'epochs' : epochs,
                        'batch_size' : batch_size,
                        'n_nodes' : 90,
                        'on_treshold' : 2000
                    }
                },
            }),
            "MLP" : MLP( {
                "verbose" : 2,
                "training_history_folder" : base_path + "history/MLP/",
                "results_folder" : base_path + "results/MLP/",
                "checkpoint_folder" : base_path + "temp_weights/MLP/",
                "plots_folder" : base_path + "plots/MLP/",
                #"load_model_folder" : base_path + "temp_weights/MLP/",
                "appliances" : {
                    "kettle" : {
                        'timewindow' : timestep*sequence_length,
                        'timestep' : timestep,
                        'overlap' :  timestep*sequence_length - timestep,
                        'epochs' : epochs,
                        'batch_size' : batch_size,
                        'feature_extractor' : "wt",
                        'on_treshold' : 2000
                    }
                },
            }),
            "MLP_Raw" : MLP( {
                "verbose" : 2,
                "training_history_folder" : base_path + "history/MLP_Raw/",
                "results_folder" : base_path + "results/MLP_Raw/",
                "checkpoint_folder" : base_path + "temp_weights/MLP_Raw/",
                "plots_folder" : base_path + "plots/MLP_Raw/",
                #"load_model_folder" : base_path + "temp_weights/MLP_Raw/",
                "appliances" : {
                    "kettle" : {
                        'timewindow' : timestep*sequence_length,
                        'timestep' : timestep,
                        'overlap' :  timestep*sequence_length - timestep,
                        'epochs' : epochs,
                        'batch_size' : batch_size,
                        'on_treshold' : 2000
                    }
                },
            }),
        },
        'train': {    
            'datasets': {
                'UKDale': {
                    'path': '../../../../../datasets/ukdale/ukdale.h5',
                    'buildings': {
                        3: {
                            'start_time': "2013-03-01",
                            'end_time' : "2013-04-09"
                        },
                        2: {
                            'start_time': "2013-04-17",
                            'end_time': "2013-10-09",
                        },             
                    }
                },
            }
        },
        'test': {
            'datasets': {
                'UKDale': {
                    'path': '../../../../../datasets/ukdale/ukdale.h5',
                    'buildings': {
                        5: {
                            'start_time': "2014-08-01",
                            'end_time': "2014-09-01"
                        }
                    }
                },
            },
            'metrics':['mae', 'rmse']
        }
    }
    ### Training and testing kettle ####
    results = API(kettle)

    #Get all the results in the experiment and print them.
    errors_keys = results.errors_keys
    errors = results.errors

    for app in results.appliances:
        f = open(base_path + "results_" + app.replace(" ", "_") + ".txt", "w")
        for classifier in errors[0].columns:
            f.write(5*"-" + classifier + "-"*5 + "\n")
            for i in range(len(errors)):
                f.write(errors_keys[i].split("_")[-1].upper() + " : " + str(errors[i][classifier][app]) + "\n")

    ###################################################
    for m in results.methods:
        create_path(base_path + "models/" + m)
        results.methods[m].save_model(base_path + "models/" + m)

def run_microwave(base_path, timestep, epochs, batch_size, sequence_length):
    microwave = {
        'power': {'mains': ['active'],'appliance': ['active']},
        'sample_rate': timestep,
        'appliances': ['microwave', 'washing machine', 'dish washer'],
        'methods': {
            'DAE':DAE({
                'n_epochs':epochs,
                'batch_size' : batch_size,
                'sequence_length': sequence_length,
                "training_history_folder" : base_path + "history/DAE/",
                "plots_folder" : base_path + "plots/DAE/",
                "file_prefix" : base_path + "temp_weights/DAE/",
                "on_threshold" : 200
            }),
            'Seq2Point':Seq2Point({
                'n_epochs':epochs,
                'batch_size' : batch_size,
                'sequence_length': sequence_length,
                "training_history_folder" : base_path + "history/Seq2Point/",
                "plots_folder" : base_path + "plots/Seq2Point/",
                "file_prefix" : base_path + "temp_weights/Seq2Point/",
                "on_threshold" : 200
            }),
            'Seq2Seq':Seq2Seq({
                'n_epochs':epochs,
                'sequence_length': sequence_length,
                'batch_size' : batch_size,
                "training_history_folder" : base_path + "history/Seq2Seq/",
                "plots_folder" : base_path + "plots/Seq2Seq/",
                "file_prefix" : base_path + "temp_weights/Seq2Seq/",
                "on_threshold" : 200
            }),
            "ResNet" : ResNet( {
                "verbose" : 2,
                "training_history_folder" : base_path + "history/ResNet/",
                "results_folder" : base_path + "results/ResNet/",
                "checkpoint_folder" : base_path + "temp_weights/ResNet/",
                "plots_folder" : base_path + "plots/ResNet/",
                #"load_model_folder" : base_path + "temp_weights/ResNet/",
                "appliances" : {
                    "microwave" : {
                        'timewindow' : timestep*sequence_length,
                        'timestep' : timestep,
                        'overlap' :  timestep*sequence_length - timestep,
                        'epochs' : epochs,
                        'batch_size' : batch_size,
                        'n_nodes' : 32,
                        'on_treshold' : 200
                    },
                },
            }),
            "DeepGRU" : DeepGRU({
                'verbose' : 2,
                "training_history_folder" : base_path + "history/DeepGRU/",
                "results_folder" : base_path + "results/DeepGRU/",
                "checkpoint_folder" : base_path + "temp_weights/DeepGRU/",
                "plots_folder" : base_path + "plots/DeepGRU/",
                #"load_model_folder" : base_path + "temp_weights/DeepGRU/",
                "appliances" : {
                    "microwave" : {
                        'timewindow' : timestep*sequence_length,
                        'timestep' : timestep,
                        'overlap' :  timestep*sequence_length - timestep,
                        'epochs' : epochs,
                        'batch_size' : batch_size,
                        'n_nodes' : 90,
                        'on_treshold' : 200
                    },
                },
            }),
            "MLP" : MLP( {
                "verbose" : 2,
                "training_history_folder" : base_path + "history/MLP/",
                "results_folder" : base_path + "results/MLP/",
                "checkpoint_folder" : base_path + "temp_weights/MLP/",
                "plots_folder" : base_path + "plots/MLP/",
                #"load_model_folder" : base_path + "temp_weights/MLP/",
                "appliances" : {
                    "microwave" : {
                        'timewindow' : timestep*sequence_length,
                        'timestep' : timestep,
                        'overlap' :  timestep*sequence_length - timestep,
                        'epochs' : epochs,
                        'batch_size' : batch_size,
                        'feature_extractor' : "wt",
                        'on_treshold' : 200
                    },
                },
            }),
            "MLP_Raw" : MLP( {
                "verbose" : 2,
                "training_history_folder" : base_path + "history/MLP_Raw/",
                "results_folder" : base_path + "results/MLP_Raw/",
                "checkpoint_folder" : base_path + "temp_weights/MLP_Raw/",
                "plots_folder" : base_path + "plots/MLP_Raw/",
                #"load_model_folder" : base_path + "temp_weights/MLP_Raw/",
                "appliances" : {
                    "microwave" : {
                        'timewindow' : timestep*sequence_length,
                        'timestep' : timestep,
                        'overlap' :  timestep*sequence_length - timestep,
                        'epochs' : epochs,
                        'batch_size' : batch_size,
                        'on_treshold' : 200
                    }
                }
            }),
        },
        'train': {    
            'datasets': {
                'UKDale': {
                    'path': '../../../../../datasets/ukdale/ukdale.h5',
                    'buildings': {
                        1: {
                            'start_time': "2013-04-17",
                            'end_time': "2013-10-09",
                        },
                        2: {
                            'start_time': "2013-04-17",
                            'end_time': "2013-10-09",
                        }           
                    }
                },
            }
        },
        'test': {
            'datasets': {
                'UKDale': {
                    'path': '../../../../../datasets/ukdale/ukdale.h5',
                    'buildings': {
                        5: {
                            'start_time': "2014-08-01",
                            'end_time': "2014-09-01"
                        } 
                    }
                },
            },
            'metrics':['mae', 'rmse']
        }
    }

    ### Training and testing microwave ####
    results = API(microwave)

    #Get all the results in the experiment and print them.
    errors_keys = results.errors_keys
    errors = results.errors

    for app in results.appliances:
        f = open(base_path + "results_" + app.replace(" ", "_") + ".txt", "w")
        for classifier in errors[0].columns:
            f.write(5*"-" + classifier + "-"*5 + "\n")
            for i in range(len(errors)):
                f.write(errors_keys[i].split("_")[-1].upper() + " : " + str(errors[i][classifier][app]) + "\n")#

    ###################################################
    for m in results.methods:
        create_path(base_path + "models/" + m)
        results.methods[m].save_model(base_path + "models/" + m)

def run_dish_washer(base_path, timestep, epochs, batch_size, sequence_length):
    dish_washer = {
        'power': {'mains': ['active'],'appliance': ['active']},
        'sample_rate': timestep,
        'appliances': ['dish washer'],
        'methods': {
            'DAE':DAE({
                'n_epochs':epochs,
                'batch_size' : batch_size,
                'sequence_length': sequence_length,
                "training_history_folder" : base_path + "history/DAE/",
                "plots_folder" : base_path + "plots/DAE/",
                "file_prefix" : base_path + "temp_weights/DAE/",
                "on_threshold" : 50
            }),
            'Seq2Point':Seq2Point({
                'n_epochs':epochs,
                'batch_size' : batch_size,
                'sequence_length': sequence_length,
                "training_history_folder" : base_path + "history/Seq2Point/",
                "plots_folder" : base_path + "plots/Seq2Point/",
                "file_prefix" : base_path + "temp_weights/Seq2Point/",
                "on_threshold" : 50
            }),
            'Seq2Seq':Seq2Seq({
                'n_epochs':epochs,
                'sequence_length': sequence_length,
                'batch_size' : batch_size,
                "training_history_folder" : base_path + "history/Seq2Seq/",
                "plots_folder" : base_path + "plots/Seq2Seq/",
                "file_prefix" : base_path + "temp_weights/Seq2Seq/",
                "on_threshold" : 50
            }),
            "ResNet" : ResNet( {
                "verbose" : 2,
                "training_history_folder" : base_path + "history/ResNet/",
                "results_folder" : base_path + "results/ResNet/",
                "checkpoint_folder" : base_path + "temp_weights/ResNet/",
                "plots_folder" : base_path + "plots/ResNet/",
                #"load_model_folder" : base_path + "temp_weights/ResNet/",
                "appliances" : {
                    "dish washer" : {
                        'timewindow' : timestep*sequence_length,
                        'timestep' : timestep,
                        'overlap' :  timestep*sequence_length - timestep,
                        'epochs' : epochs,
                        'batch_size' : batch_size,
                        'n_nodes' : 32,
                        "on_threshold" : 50
                    }
                },
            }),
            "DeepGRU" : DeepGRU({
                'verbose' : 2,
                "training_history_folder" : base_path + "history/DeepGRU/",
                "results_folder" : base_path + "results/DeepGRU/",
                "checkpoint_folder" : base_path + "temp_weights/DeepGRU/",
                "plots_folder" : base_path + "plots/DeepGRU/",
                #"load_model_folder" : base_path + "temp_weights/DeepGRU/",
                "appliances" : {
                    "dish washer" : {
                        'timewindow' : timestep*sequence_length,
                        'timestep' : timestep,
                        'overlap' :  timestep*sequence_length - timestep,
                        'epochs' : epochs,
                        'batch_size' : batch_size,
                        'n_nodes' : 90,
                        'on_treshold' : 50
                    }
                },
            }),
            "MLP" : MLP( {
                "verbose" : 2,
                "training_history_folder" : base_path + "history/MLP/",
                "results_folder" : base_path + "results/MLP/",
                "checkpoint_folder" : base_path + "temp_weights/MLP/",
                "plots_folder" : base_path + "plots/MLP/",
                #"load_model_folder" : base_path + "temp_weights/MLP/",
                "appliances" : {
                    "dish washer" : {
                        'timewindow' : timestep*sequence_length,
                        'timestep' : timestep,
                        'overlap' :  timestep*sequence_length - timestep,
                        'epochs' : epochs,
                        'batch_size' : batch_size,
                        'feature_extractor' : "wt",
                        'on_treshold' : 50
                    },
                },
            }),
            "MLP_Raw" : MLP( {
                "verbose" : 2,
                "training_history_folder" : base_path + "history/MLP_Raw/",
                "results_folder" : base_path + "results/MLP_Raw/",
                "checkpoint_folder" : base_path + "temp_weights/MLP_Raw/",
                "plots_folder" : base_path + "plots/MLP_Raw/",
                #"load_model_folder" : base_path + "temp_weights/MLP_Raw/",
                "appliances" : {
                    "dish washer" : {
                        'timewindow' : timestep*sequence_length,
                        'timestep' : timestep,
                        'overlap' :  timestep*sequence_length - timestep,
                        'epochs' : epochs,
                        'batch_size' : batch_size,
                        'on_treshold' : 50
                    },  
                },
            }),
        },
        'train': {    
            'datasets': {
                'UKDale': {
                    'path': '../../../../../datasets/ukdale/ukdale.h5',
                    'buildings': {
                        1: {
                            'start_time': "2013-04-17",
                            'end_time': "2013-10-09",
                        },
                        2: {
                            'start_time': "2013-04-17",
                            'end_time': "2013-10-09",
                        }           
                    }
                },
            }
        },
        'test': {
            'datasets': {
                'UKDale': {
                    'path': '../../../../../datasets/ukdale/ukdale.h5',
                    'buildings': {
                        5: {
                            'start_time': "2014-08-01",
                            'end_time': "2014-09-01"
                        } 
                    }
                },
            },
            'metrics':['mae', 'rmse']
        }
    }

    ### Training and testing dish washer ####
    results = API(dish_washer)

    #Get all the results in the experiment and print them.
    errors_keys = results.errors_keys
    errors = results.errors

    for app in results.appliances:
        f = open(base_path + "results_" + app.replace(" ", "_") + ".txt", "w")
        for classifier in errors[0].columns:
            f.write(5*"-" + classifier + "-"*5 + "\n")
            for i in range(len(errors)):
                f.write(errors_keys[i].split("_")[-1].upper() + " : " + str(errors[i][classifier][app]) + "\n")#

    ###################################################
    for m in results.methods:
        create_path(base_path + "models/" + m)
        results.methods[m].save_model(base_path + "models/" + m)

def run_washing_machine(base_path, timestep, epochs, batch_size, sequence_length):
    washing_machine = {
        'power': {'mains': ['active'],'appliance': ['active']},
        'sample_rate': timestep,
        'appliances': ['washing machine'],
        'methods': {
            'DAE':DAE({
                'n_epochs':epochs,
                'batch_size' : batch_size,
                'sequence_length': sequence_length,
                "training_history_folder" : base_path + "history/DAE/",
                "plots_folder" : base_path + "plots/DAE/",
                "file_prefix" : base_path + "temp_weights/DAE/",
                "on_threshold" : 50
            }),
            'Seq2Point':Seq2Point({
                'n_epochs':epochs,
                'batch_size' : batch_size,
                'sequence_length': sequence_length,
                "training_history_folder" : base_path + "history/Seq2Point/",
                "plots_folder" : base_path + "plots/Seq2Point/",
                "file_prefix" : base_path + "temp_weights/Seq2Point/",
                "on_threshold" : 50
            }),
            'Seq2Seq':Seq2Seq({
                'n_epochs':epochs,
                'sequence_length': sequence_length,
                'batch_size' : batch_size,
                "training_history_folder" : base_path + "history/Seq2Seq/",
                "plots_folder" : base_path + "plots/Seq2Seq/",
                "file_prefix" : base_path + "temp_weights/Seq2Seq/",
                "on_threshold" : 50
            }),
            "ResNet" : ResNet( {
                "verbose" : 2,
                "training_history_folder" : base_path + "history/ResNet/",
                "results_folder" : base_path + "results/ResNet/",
                "checkpoint_folder" : base_path + "temp_weights/ResNet/",
                "plots_folder" : base_path + "plots/ResNet/",
                #"load_model_folder" : base_path + "temp_weights/ResNet/",
                "appliances" : {
                    "washing machine" : {
                        'timewindow' : timestep*sequence_length,
                        'timestep' : timestep,
                        'overlap' :  timestep*sequence_length - timestep,
                        'epochs' : epochs,
                        'batch_size' : batch_size,
                        'n_nodes' : 32,
                        'on_treshold' : 50
                    }
                },
            }),
            "DeepGRU" : DeepGRU({
                'verbose' : 2,
                "training_history_folder" : base_path + "history/DeepGRU/",
                "results_folder" : base_path + "results/DeepGRU/",
                "checkpoint_folder" : base_path + "temp_weights/DeepGRU/",
                "plots_folder" : base_path + "plots/DeepGRU/",
                #"load_model_folder" : base_path + "temp_weights/DeepGRU/",
                "appliances" : {
                    "washing machine" : {
                        'timewindow' : timestep*sequence_length,
                        'timestep' : timestep,
                        'overlap' :  timestep*sequence_length - timestep,
                        'epochs' : epochs,
                        'batch_size' : batch_size,
                        'n_nodes' : 90,
                        'on_treshold' : 50
                    }
                },
            }),
            "MLP" : MLP( {
                "verbose" : 2,
                "training_history_folder" : base_path + "history/MLP/",
                "results_folder" : base_path + "results/MLP/",
                "checkpoint_folder" : base_path + "temp_weights/MLP/",
                "plots_folder" : base_path + "plots/MLP/",
                #"load_model_folder" : base_path + "temp_weights/MLP/",
                "appliances" : {
                    "washing machine" : {
                        'timewindow' : timestep*sequence_length,
                        'timestep' : timestep,
                        'overlap' :  timestep*sequence_length - timestep,
                        'epochs' : epochs,
                        'batch_size' : batch_size,
                        'feature_extractor' : "wt",
                        'on_treshold' : 50
                    }
                },
            }),
            "MLP_Raw" : MLP( {
                "verbose" : 2,
                "training_history_folder" : base_path + "history/MLP_Raw/",
                "results_folder" : base_path + "results/MLP_Raw/",
                "checkpoint_folder" : base_path + "temp_weights/MLP_Raw/",
                "plots_folder" : base_path + "plots/MLP_Raw/",
                #"load_model_folder" : base_path + "temp_weights/MLP_Raw/",
                "appliances" : {
                    "washing machine" : {
                        'timewindow' : timestep*sequence_length,
                        'timestep' : timestep,
                        'overlap' :  timestep*sequence_length - timestep,
                        'epochs' : epochs,
                        'batch_size' : batch_size,
                        'on_treshold' : 50
                    }     
                },
            }),
        },
        'train': {    
            'datasets': {
                'UKDale': {
                    'path': '../../../../../datasets/ukdale/ukdale.h5',
                    'buildings': {
                        1: {
                            'start_time': "2013-04-17",
                            'end_time': "2013-10-09",
                        },
                        2: {
                            'start_time': "2013-04-17",
                            'end_time': "2013-10-09",
                        }           
                    }
                },
            }
        },
        'test': {
            'datasets': {
                'UKDale': {
                    'path': '../../../../../datasets/ukdale/ukdale.h5',
                    'buildings': {
                        5: {
                            'start_time': "2014-08-01",
                            'end_time': "2014-09-01"
                        } 
                    }
                },
            },
            'metrics':['mae', 'rmse']
        }
    }

    ### Training and testing washing machine ####
    results = API(washing_machine)

    #Get all the results in the experiment and print them.
    errors_keys = results.errors_keys
    errors = results.errors

    for app in results.appliances:
        f = open(base_path + "results_" + app.replace(" ", "_") + ".txt", "w")
        for classifier in errors[0].columns:
            f.write(5*"-" + classifier + "-"*5 + "\n")
            for i in range(len(errors)):
                f.write(errors_keys[i].split("_")[-1].upper() + " : " + str(errors[i][classifier][app]) + "\n")#

    ###################################################
    for m in results.methods:
        create_path(base_path + "models/" + m)
        results.methods[m].save_model(base_path + "models/" + m)

if __name__ == "__main__":
    #base_path= "/home/rteixeira/soa_results/"
    #base_path = "/home/user/article_results/"
    base_path = "/home/atnoguser/transfer_results/ukdale/"
    epochs = 1
    batch_size = 256
    sequence_length = 299
    timestep = 6

    run_fridge(base_path, timestep, epochs, batch_size, sequence_length)
    #run_microwave(base_path, timestep, epochs, batch_size, sequence_length)
    #run_dish_washer(base_path, timestep, epochs, batch_size, sequence_length)
    #run_kettle(base_path, timestep, epochs, batch_size, sequence_length)
    #run_washing_machine(base_path, timestep, epochs, batch_size, sequence_length)