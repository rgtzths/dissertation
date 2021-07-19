from nilmtk.api import API

import sys
sys.path.insert(1, "../../regression_models")
sys.path.insert(1, "../../utils")
sys.path.insert(1, "../../feature_extractors")

from resnet import ResNet
from deep_gru import DeepGRU
from mlp_dwt import MLP

def run_fridge(base_path, timewindow, timestep, epochs):
    fridge = {
        'power': {'mains': ['apparent', 'active', 'reactive'],'appliance': ['active']},
        'voltage' : {'mains' : ['active']},
        'power factor' : {'mains' : ['active']},
        'frequency' : {'mains' : ['active']},
        'current' : {'mains' : ['active']},
        'cumulative energy' : {'mains' : ['apparent', 'active', 'reactive']},
        'sample_rate': timestep,
        'appliances': ['fridge'],
        'methods': {
            "ResNet" : ResNet( {
                "verbose" : 2,
                "training_history_folder" : base_path + "history/ResNet/",
                "results_folder" : base_path + "results/ResNet/",
                "checkpoint_folder" : base_path + "models/ResNet/",
                "plots_folder" : base_path + "plots/ResNet/",
                #"load_model_folder" : base_path + "models/ResNet/",
                "appliances" : {
                    "fridge" : {
                        'timewindow' : timewindow,
                        'timestep' : timestep,
                        'overlap' : timewindow-timestep,
                        'epochs' : epochs,
                        'batch_size' : 1024,
                        'n_nodes' : 32
                    }
                },
                "predicted_column": ("power", "active"),
            }),
            "DeepGRU" : DeepGRU({
                'verbose' : 2,
                "training_history_folder" : base_path + "history/DeepGRU/",
                "results_folder" : base_path + "results/DeepGRU/",
                "checkpoint_folder" : base_path + "models/DeepGRU/",
                "plots_folder" : base_path + "plots/DeepGRU/",
                #"load_model_folder" : base_path + "models/DeepGRU/",
                "appliances" : {
                    "fridge" : {
                        'timewindow' : timewindow,
                        'timestep' : timestep,
                        'overlap' : timewindow-timestep,
                        'epochs' : epochs,
                        'batch_size' : 1024,
                        'n_nodes' : 90
                    }
                },
                "predicted_column": ("power", "active"),
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
                        'timewindow' : timewindow,
                        'timestep' : timestep,
                        'overlap' : timewindow-timestep,
                        'epochs' : epochs,
                        'batch_size' : 1024,
                        'feature_extractor' : "dwt"
                    }
                },
                "predicted_column": ("power", "active"), 
            }),
        },
        #'train': {    
        #    'datasets': {
        #        'withus': {
        #            'path': '../../../datasets/withus_h5/withus.h5',
        #            'buildings': {
        #                1: {
        #                    'start_time': "2021-04-01",
        #                    'end_time' : "2021-05-01"
        #                },
        #                2: {
        #                    'start_time': "2021-04-01",
        #                    'end_time' : "2021-05-01"
        #                }        
        #            }
        #        },
        #    }
        #},
        #'test': {
        #    'datasets': {
        #        'withus': {
        #            'path': '../../../datasets/withus_h5/withus.h5',
        #            'buildings': {
        #                3: {
        #                    'start_time': "2021-05-01",
        #                    'end_time' : "2021-06-01"
        #                }
        #            }
        #        }
        #    },
        #    'metrics':['mae', 'rmse']
        #}
        'train': {    
            'datasets': {
                'withus': {
                    'path': '../../../datasets/withus_h5/withus.h5',
                    'buildings': {
                        1: {
                            'start_time': "2021-04-01",
                            'end_time' : "2021-04-02"
                        },          
                    }
                },
            }
        },
        'test': {
            'datasets': {
                'withus': {
                    'path': '../../../datasets/withus_h5/withus.h5',
                    'buildings': {
                        2: {
                            'start_time': "2021-05-01",
                            'end_time': "2021-05-02"
                        } 
                    }
                }
            },
            'metrics':['mae', 'rmse']
        }
    }

    ### Training and testing Fridge ####
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

    ##################################################

def run_heat_pump(base_path, timewindow, timestep, epochs):
    heat_pump = {
        'power': {'mains': ['apparent', 'active', 'reactive'],'appliance': ['active']},
        'voltage' : {'mains' : ['active']},
        'power factor' : {'mains' : ['active']},
        'frequency' : {'mains' : ['active']},
        'current' : {'mains' : ['active']},
        'cumulative energy' : {'mains' : ['apparent', 'active', 'reactive']},
        'sample_rate' : timestep,
        'appliances': ['heat pump'],
        'methods': {
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
                        'overlap' : timewindow-timestep,
                        'epochs' : epochs,
                        'batch_size' : 1024,
                        'n_nodes' : 32
                    }
                },
                "predicted_column": ("power", "active"),
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
                        'overlap' : timewindow-timestep,
                        'epochs' : epochs,
                        'batch_size' : 1024,
                        'n_nodes' : 90
                    }
                },
                "predicted_column": ("power", "active"),
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
                        'overlap' : timewindow-timestep,
                        'epochs' : epochs,
                        'batch_size' : 1024,
                        'feature_extractor' : "dwt"
                    }
                },
                "predicted_column": ("power", "active"), 
            }),
        },
        #'train': {    
        #    'datasets': {
        #        'withus': {
        #            'path': '../../../datasets/withus_h5/withus.h5',
        #            'buildings': {
        #                2: {
        #                    'start_time': "2021-03-23",
        #                    'end_time' : "2021-05-22"
        #                },           
        #            }
        #        },
        #    }
        #},
        #'test': {
        #    'datasets': {
        #        'withus': {
        #            'path': '../../../datasets/withus_h5/withus.h5',
        #            'buildings': {
        #                2: {
        #                    'start_time' : "2021-06-01",
        #                    'end_time' : "2021-07-01"
        #                }
        #            }
        #        },
        #    },
        #    'metrics':['mae', 'rmse']
        #}
        'train': {    
            'datasets': {
                'withus': {
                    'path': '../../../datasets/withus_h5/withus.h5',
                    'buildings': {
                        2: {
                            'start_time': "2021-03-23",
                            'end_time' : "2021-03-24"
                        },           
                    }
                },
            }
        },
        'test': {
            'datasets': {
                'withus': {
                    'path': '../../../datasets/withus_h5/withus.h5',
                    'buildings': {
                        2: {
                            'start_time' : "2021-04-24",
                            'end_time' : "2021-04-25"
                        }
                    }
                },
            },
            'metrics':['mae', 'rmse']
        }
    }
    ### Training and testing heat pump ####
    results = API(heat_pump)

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

if __name__ == "__main__":
    #output_path= "/home/rteixeira/temporary/regression/sampling/"
    output_path = "/home/user/thesis_results/regression/sampling/"
    epochs = 1
    timesteps = [2, 60]
    timewindow = 50

    for timestep in timesteps:

        run_fridge(output_path + str(timestep) + "/", timewindow*timestep, timestep, epochs)
        run_heat_pump(output_path + str(timestep) + "/", timewindow*timestep, timestep, epochs)