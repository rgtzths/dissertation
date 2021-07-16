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
        'train': {    
            'datasets': {
                'withus': {
                    'path': '../../../datasets/withus_h5/withus.h5',
                    'buildings': {
                        1: {
                            'start_time': "2021-04-01",
                            'end_time' : "2021-05-01"
                        },
                        2: {
                            'start_time': "2021-04-01",
                            'end_time' : "2021-05-01"
                        }           
                    }
                },
            }
        },
        'test': {
            'datasets': {
                'withus': {
                    'path': '../../../datasets/withus_h5/withus.h5',
                    'buildings': {
                        3: {
                            'start_time': "2021-05-01",
                            'end_time': "2021-06-01"
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
        'train': {    
            'datasets': {
                'withus': {
                    'path': '../../../datasets/withus_h5/withus.h5',
                    'buildings': {
                        2: {
                            'start_time': "2021-03-23",
                            'end_time' : "2021-04-23"
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
                            'end_time' : "2021-05-22"
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

def run_dish_washer(base_path, timewindow, timestep, epochs):
    dish_washer = {
        'power': {'mains': ['apparent', 'active', 'reactive'],'appliance': ['active']},
        'voltage' : {'mains' : ['active']},
        'power factor' : {'mains' : ['active']},
        'frequency' : {'mains' : ['active']},
        'current' : {'mains' : ['active']},
        'cumulative energy' : {'mains' : ['apparent', 'active', 'reactive']},
        'sample_rate' : timestep,
        'appliances': ['dish washer'],
        'methods': {
            "ResNet" : ResNet( {
                "verbose" : 2,
                "training_history_folder" : base_path + "history/ResNet/",
                "results_folder" : base_path + "results/ResNet/",
                "checkpoint_folder" : base_path + "models/ResNet/",
                "plots_folder" : base_path + "plots/ResNet/",
                #"load_model_folder" : base_path + "models/ResNet/",
                "appliances" : {
                    "dish washer" : {
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
                    "dish washer" : {
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
                    "dish washer" : {
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
        'train': {    
            'datasets': {
                'withus': {
                    'path': '../../../datasets/withus_h5/withus.h5',
                    'buildings': {
                        3: {
                            'start_time': "2021-03-13",
                            'end_time' : "2021-03-21"
                        },          
                    }
                },
                'withus2': {
                    'path': '../../../datasets/withus_h5/withus.h5',
                    'buildings': {
                        3: {
                            'start_time': "2021-03-22",
                            'end_time' : "2021-04-17"
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
                        3: {
                            'start_time': "2021-04-19",
                            'end_time' : "2021-05-19"
                        },          
                    }
                },
            },
            'metrics':['mae', 'rmse']
        }
    }

    ### Training and testing Dish Washer ####
    results = API(dish_washer)

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

def run_washing_machine(base_path, timewindow, timestep, epochs):
    washing_machine = {
        'power': {'mains': ['apparent', 'active', 'reactive'],'appliance': ['active']},
        'voltage' : {'mains' : ['active']},
        'power factor' : {'mains' : ['active']},
        'frequency' : {'mains' : ['active']},
        'current' : {'mains' : ['active']},
        'cumulative energy' : {'mains' : ['apparent', 'active', 'reactive']},
        'sample_rate' : timestep,
        'appliances': ['washing machine'],
        'methods': {
            "ResNet" : ResNet( {
                "verbose" : 2,
                "training_history_folder" : base_path + "history/ResNet/",
                "results_folder" : base_path + "results/ResNet/",
                "checkpoint_folder" : base_path + "models/ResNet/",
                "plots_folder" : base_path + "plots/ResNet/",
                #"load_model_folder" : base_path + "models/ResNet/",
                "appliances" : {
                    "washing machine" : {
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
                    "washing machine" : {
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
                    "washing machine" : {
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
        'train': {    
            'datasets': {
                'withus': {
                    'path': '../../../datasets/withus_h5/withus.h5',
                    'buildings': {
                        3: {
                            'start_time': "2021-03-13",
                            'end_time' : "2021-03-15"
                        }            
                    }
                },
                'withus2': {
                    'path': '../../../datasets/withus_h5/withus.h5',
                    'buildings': {
                        3: {
                            'start_time': "2021-03-16",
                            'end_time' : "2021-03-17"
                        }            
                    }
                },
                'withus3': {
                    'path': '../../../datasets/withus_h5/withus.h5',
                    'buildings': {
                        3: {
                            'start_time': "2021-03-19",
                            'end_time' : "2021-03-20"
                        }         
                    }
                },
                'withus4': {
                    'path': '../../../datasets/withus_h5/withus.h5',
                    'buildings': {
                        3: {
                            'start_time': "2021-03-22",
                            'end_time' : "2021-03-23"
                        }             
                    }
                },
                'withus5': {
                    'path': '../../../datasets/withus_h5/withus.h5',
                    'buildings': {
                        3: {
                            'start_time': "2021-03-26",
                            'end_time' : "2021-03-29"
                        }           
                    }
                },
                'withus6': {
                    'path': '../../../datasets/withus_h5/withus.h5',
                    'buildings': {
                        3: {
                            'start_time': "2021-03-30",
                            'end_time' : "2021-04-01"
                        }            
                    }
                },
                'withus7': {
                    'path': '../../../datasets/withus_h5/withus.h5',
                    'buildings': {
                        3: {
                            'start_time': "2021-04-02",
                            'end_time' : "2021-04-04"
                        }              
                    }
                },
                'withus8': {
                    'path': '../../../datasets/withus_h5/withus.h5',
                    'buildings': {
                        3: {
                            'start_time': "2021-04-05",
                            'end_time' : "2021-04-07"
                        }           
                    }
                },
                'withus9': {
                    'path': '../../../datasets/withus_h5/withus.h5',
                    'buildings': {
                        3: {
                            'start_time': "2021-04-08",
                            'end_time' : "2021-04-09"
                        }  
                    }
                },
                'withus10': {
                    'path': '../../../datasets/withus_h5/withus.h5',
                    'buildings': {
                        3: {
                            'start_time': "2021-04-11",
                            'end_time' : "2021-04-12"
                        }             
                    }
                },
                'withus11': {
                    'path': '../../../datasets/withus_h5/withus.h5',
                    'buildings': {
                        3: {
                            'start_time': "2021-04-13",
                            'end_time' : "2021-04-14"
                        }             
                    }
                },
                'withus12': {
                    'path': '../../../datasets/withus_h5/withus.h5',
                    'buildings': {
                        3: {
                            'start_time': "2021-04-16",
                            'end_time' : "2021-04-17"
                        }              
                    }
                },
                'withus13': {
                    'path': '../../../datasets/withus_h5/withus.h5',
                    'buildings': {
                        3: {
                            'start_time': "2021-04-19",
                            'end_time' : "2021-04-20"
                        }              
                    }
                },
                'withus14': {
                    'path': '../../../datasets/withus_h5/withus.h5',
                    'buildings': {
                        3: {
                            'start_time': "2021-04-23",
                            'end_time' : "2021-04-25"
                        }            
                    }
                },
                'withus15': {
                    'path': '../../../datasets/withus_h5/withus.h5',
                    'buildings': {
                        3: {
                            'start_time': "2021-04-26",
                            'end_time' : "2021-04-27"
                        }              
                    }
                },
                'withus16': {
                    'path': '../../../datasets/withus_h5/withus.h5',
                    'buildings': {
                        3: {
                            'start_time': "2021-04-29",
                            'end_time' : "2021-05-01"
                        }              
                    }
                },
                'withus17': {
                    'path': '../../../datasets/withus_h5/withus.h5',
                    'buildings': {
                        3: {
                            'start_time': "2021-05-03",
                            'end_time' : "2021-05-04"
                        }       
                    }
                },
                'withus18': {
                    'path': '../../../datasets/withus_h5/withus.h5',
                    'buildings': {
                        3: {
                            'start_time': "2021-05-05",
                            'end_time' : "2021-05-06"
                        }               
                    }
                },
                'withus19': {
                    'path': '../../../datasets/withus_h5/withus.h5',
                    'buildings': {
                        3: {
                            'start_time': "2021-05-07",
                            'end_time' : "2021-05-08"
                        }                  
                    }
                },
                'withus20': {
                    'path': '../../../datasets/withus_h5/withus.h5',
                    'buildings': {
                        3: {
                            'start_time': "2021-05-14",
                            'end_time' : "2021-05-17"
                        }                  
                    }
                }
            }
        },
        'test': {
            'datasets': {
                'withus': {
                    'path': '../../../datasets/withus_h5/withus.h5',
                    'buildings': {
                        3: {
                            'start_time': "2021-05-06",
                            'end_time' : "2021-06-06"
                        } 
                    }
                },
            },
            'metrics':['mae', 'rmse']
        }
    }

    ### Training and testing Washing Machine ####
    results = API(washing_machine)

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
    output_path= "/home/rteixeira/temporary/regression/withus_all_features/"
    #base_path = "/home/user/thesis_results/regression/"
    epochs = 1
    timesteps = [2, 60]
    timewindow = 50

    for timestep in timesteps:

        run_fridge(output_path + str(timestep) + "/", timewindow*timestep, timestep, epochs)
        run_heat_pump(output_path + str(timestep) + "/", timewindow*timestep, timestep, epochs)
        run_dish_washer(output_path + str(timestep) + "/", timewindow*timestep, timestep, epochs)
        run_washing_machine(output_path + str(timestep) + "/", timewindow*timestep, timestep, epochs)

