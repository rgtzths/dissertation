from nilmtk.api import API

import sys
sys.path.insert(1, "../../nilmtk-contrib")
sys.path.insert(1, "../../regression_models")
sys.path.insert(1, "../../utils")
sys.path.insert(1, "../../feature_extractors")

from utils import create_path
from dae import DAE
from seq2point import Seq2Point
from seq2seq import Seq2Seq
from rnn import RNN
from WindowGRU import WindowGRU

from resnet import ResNet
from deep_gru import DeepGRU
from mlp_dwt import MLP

#base_path= "/home/rteixeira/soa_results/"
#base_path = "/home/user/article_results/"
base_path = "/home/atnoguser/transfer_results/"
epochs = 1
timestep = 6
timewindow = 64 * timestep
overlap = timewindow - timestep

# Classifies Dish Washer, Fridge, Microwave, washing machine and kettle
fridge = {
    'power': {'mains': ['apparent'],'appliance': ['active']},
    'sample_rate': timestep,
    'appliances': ['fridge'],
    'methods': {
        "ResNet" : ResNet( {
            "verbose" : 2,
            "training_history_folder" : base_path + "history/ResNet/",
            "results_folder" : base_path + "results/ResNet/",
            "checkpoint_folder" : base_path + "temp_weights/ResNet/",
            "plots_folder" : base_path + "plots/ResNet/",
            #"load_model_folder" : base_path + "temp_weights/ResNet/",
            "appliances" : {
                "fridge" : {
                    'timewindow' : timewindow,
                    'timestep' : timestep,
                    'overlap' :  overlap,
                    'epochs' : epochs,
                    'batch_size' : 256,
                    'n_nodes' : 32,
                    'on_treshold' : 50,
                }
            },
            "predicted_column": ("power", "active"),
        }),
        "DeepGRU" : DeepGRU({
            'verbose' : 2,
            "training_history_folder" : base_path + "history/DeepGRU/",
            "results_folder" : base_path + "results/DeepGRU/",
            "checkpoint_folder" : base_path + "temp_weights/DeepGRU/",
            "plots_folder" : base_path + "plots/DeepGRU/",
            #"load_model_folder" : base_path + "temp_weights/DeepGRU/",
            "appliances" : {
                "fridge" : {
                    'timewindow' : timewindow,
                    'timestep' : timestep,
                    'overlap' :  overlap,
                    'epochs' : epochs,
                    'batch_size' : 256,
                    'n_nodes' : 90,
                    'on_treshold' : 50,
                }
            },
            "predicted_column": ("power", "active"),
        }),
        "MLP" : MLP( {
            "verbose" : 2,
            "training_history_folder" : base_path + "history/MLP/",
            "results_folder" : base_path + "results/MLP/",
            "checkpoint_folder" : base_path + "temp_weights/MLP/",
            "plots_folder" : base_path + "plots/MLP/",
            #"load_model_folder" : base_path + "temp_weights/MLP/",
            "appliances" : {
                "fridge" : {
                    'timewindow' : timewindow,
                    'timestep' : timestep,
                    'overlap' :  overlap,
                    'epochs' : epochs,
                    'batch_size' : 256,
                    'feature_extractor' : "wt",
                    'on_treshold' : 50
                }
            },
            "predicted_column": ("power", "active"), 
        }),
        "MLP_Raw" : MLP( {
            "verbose" : 2,
            "training_history_folder" : base_path + "history/MLP_Raw/",
            "results_folder" : base_path + "results/MLP_Raw/",
            "checkpoint_folder" : base_path + "temp_weights/MLP_Raw/",
            "plots_folder" : base_path + "plots/MLP_Raw/",
            #"load_model_folder" : base_path + "temp_weights/MLP_Raw/",
            "appliances" : {
                "fridge" : {
                    'timewindow' : timewindow,
                    'timestep' : timestep,
                    'overlap' :  overlap,
                    'epochs' : epochs,
                    'batch_size' : 256,
                    'on_treshold' : 50,
                }     
            },
            "mains_mean" : 594.9043,
            "mains_std" : 513.311,
            "predicted_column": ("power", "active"), 
        }),
    },
    #'train': {    
    #    'datasets': {
    #        'UKDale': {
    #            'path': '../../../../datasets/ukdale/ukdale.h5',
    #            'buildings': {
    #                1: {
    #                    'start_time': "2013-04-17",
    #                    'end_time': "2013-07-17",
    #                },
    #                2: {
    #                    'start_time': "2013-04-17",
    #                    'end_time': "2013-07-17",
    #                }           
    #            }
    #        },
    #    }
    #},
    #'test': {
    #    'datasets': {
    #        'UKDale': {
    #            'path': '../../../../datasets/ukdale/ukdale.h5',
    #            'buildings': {
    #                5: {
    #                    'start_time': "2014-07-01",
    #                    'end_time': "2014-09-01"
    #                } 
    #            }
    #        },
    #    },
    #    'metrics':['mae', 'rmse']
    #}
    'train': {    
        'datasets': {
            'UKDale': {
                'path': '../../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    1: {
                        'start_time': "2014-02-01",
                        'end_time' : "2014-02-02"
                    },
                    2: {
                        'start_time': "2013-05-20",
                        'end_time': "2013-05-21"
                    }           
                }
            },
        }
    },
    'test': {
        'datasets': {
            'UKDale': {
                'path': '../../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    5: {
                        'start_time': "2014-07-01",
                        'end_time': "2014-07-02"
                    } 
                }
            }
        },
        'metrics':['mae', 'rmse']
    }
}

kettle = {
    'power': {'mains': ['apparent'],'appliance': ['active']},
    'sample_rate': timestep,
    'appliances': ['kettle'],
    'methods': {
        "ResNet" : ResNet( {
            "verbose" : 2,
            "training_history_folder" : base_path + "history/ResNet/",
            "results_folder" : base_path + "results/ResNet/",
            "checkpoint_folder" : base_path + "temp_weights/ResNet/",
            "plots_folder" : base_path + "plots/ResNet/",
            #"load_model_folder" : base_path + "temp_weights/ResNet/",
            "appliances" : {
                "kettle" : {
                    'timewindow' : timewindow,
                    'timestep' : timestep,
                    'overlap' :  overlap,
                    'epochs' : epochs,
                    'batch_size' : 256,
                    'n_nodes' : 32,
                    'on_treshold' : 2000
                }
            },
            "predicted_column": ("power", "active"),
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
                    'timewindow' : timewindow,
                    'timestep' : timestep,
                    'overlap' :  overlap,
                    'epochs' : epochs,
                    'batch_size' : 256,
                    'n_nodes' : 90,
                    'on_treshold' : 2000
                }
            },
            "predicted_column": ("power", "active"),
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
                    'timewindow' : timewindow,
                    'timestep' : timestep,
                    'overlap' :  overlap,
                    'epochs' : epochs,
                    'batch_size' : 256,
                    'feature_extractor' : "wt",
                    'on_treshold' : 2000
                }
            },
            "predicted_column": ("power", "active"), 
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
                    'timewindow' : timewindow,
                    'timestep' : timestep,
                    'overlap' :  overlap,
                    'epochs' : epochs,
                    'batch_size' : 256,
                    'on_treshold' : 2000
                }
            },
            "predicted_column": ("power", "active"), 
        }),
    },
    'train': {    
        'datasets': {
            'UKDale': {
                'path': '../../../../datasets/ukdale/ukdale.h5',
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
                'path': '../../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    5: {
                        'start_time': "2014-07-01",
                        'end_time': "2014-09-01"
                    }
                }
            },
        },
        'metrics':['mae', 'rmse']
    }
}

# Classifies Dish Washer, Fridge, Microwave, washing machine and kettle
others = {
    'power': {'mains': ['apparent'],'appliance': ['active']},
    'sample_rate': timestep,
    'appliances': ['microwave', 'washing machine', 'dish washer'],
    'methods': {
        "ResNet" : ResNet( {
            "verbose" : 2,
            "training_history_folder" : base_path + "history/ResNet/",
            "results_folder" : base_path + "results/ResNet/",
            "checkpoint_folder" : base_path + "temp_weights/ResNet/",
            "plots_folder" : base_path + "plots/ResNet/",
            #"load_model_folder" : base_path + "temp_weights/ResNet/",
            "appliances" : {
                "microwave" : {
                    'timewindow' : timewindow,
                    'timestep' : timestep,
                    'overlap' :  overlap,
                    'epochs' : epochs,
                    'batch_size' : 256,
                    'n_nodes' : 32,
                    'on_treshold' : 200
                },
                "dish washer" : {
                    'timewindow' : timewindow,
                    'timestep' : timestep,
                    'overlap' :  overlap,
                    'epochs' : epochs,
                    'batch_size' : 256,
                    'n_nodes' : 32,
                    "on_threshold" : 50
                },
                "washing machine" : {
                    'timewindow' : timewindow,
                    'timestep' : timestep,
                    'overlap' :  overlap,
                    'epochs' : epochs,
                    'batch_size' : 256,
                    'n_nodes' : 32,
                    'on_treshold' : 50
                }
            },
            "predicted_column": ("power", "active"),
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
                    'timewindow' : timewindow,
                    'timestep' : timestep,
                    'overlap' :  overlap,
                    'epochs' : epochs,
                    'batch_size' : 256,
                    'n_nodes' : 90,
                    'on_treshold' : 200
                },
                "dish washer" : {
                    'timewindow' : timewindow,
                    'timestep' : timestep,
                    'overlap' :  overlap,
                    'epochs' : epochs,
                    'batch_size' : 256,
                    'n_nodes' : 90,
                    'on_treshold' : 50
                },
                "washing machine" : {
                    'timewindow' : timewindow,
                    'timestep' : timestep,
                    'overlap' :  overlap,
                    'epochs' : epochs,
                    'batch_size' : 256,
                    'n_nodes' : 90,
                    'on_treshold' : 50
                }
            },
            "predicted_column": ("power", "active"),
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
                    'timewindow' : timewindow,
                    'timestep' : timestep,
                    'overlap' :  overlap,
                    'epochs' : epochs,
                    'batch_size' : 256,
                    'feature_extractor' : "wt",
                    'on_treshold' : 200
                },
                "dish washer" : {
                    'timewindow' : timewindow,
                    'timestep' : timestep,
                    'overlap' :  overlap,
                    'epochs' : epochs,
                    'batch_size' : 256,
                    'feature_extractor' : "wt",
                    'on_treshold' : 50
                },
                "washing machine" : {
                    'timewindow' : timewindow,
                    'timestep' : timestep,
                    'overlap' :  overlap,
                    'epochs' : epochs,
                    'batch_size' : 256,
                    'feature_extractor' : "wt",
                    'on_treshold' : 50
                }
            },
            "predicted_column": ("power", "active"), 
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
                    'timewindow' : timewindow,
                    'timestep' : timestep,
                    'overlap' :  overlap,
                    'epochs' : epochs,
                    'batch_size' : 256,
                    'on_treshold' : 200
                },
                "dish washer" : {
                    'timewindow' : timewindow,
                    'timestep' : timestep,
                    'overlap' :  overlap,
                    'epochs' : epochs,
                    'batch_size' : 256,
                    'on_treshold' : 50
                },
                "washing machine" : {
                    'timewindow' : timewindow,
                    'timestep' : timestep,
                    'overlap' :  overlap,
                    'epochs' : epochs,
                    'batch_size' : 256,
                    'on_treshold' : 50
                }     
            },
            "mains_mean" : 594.9043,
            "mains_std" : 513.311,
            "predicted_column": ("power", "active"), 
        }),
    },
    'train': {    
        'datasets': {
            'UKDale': {
                'path': '../../../../datasets/ukdale/ukdale.h5',
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
                'path': '../../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    5: {
                        'start_time': "2014-07-01",
                        'end_time': "2014-09-01"
                    } 
                }
            },
        },
        'metrics':['mae', 'rmse']
    }
    #'train': {    
    #    'datasets': {
    #        'UKDale': {
    #            'path': '../../../../datasets/ukdale/ukdale.h5',
    #            'buildings': {
    #                1: {
    #                    'start_time': "2014-02-01",
    #                    'end_time' : "2014-02-02"
    #                },
    #                2: {
    #                    'start_time': "2013-05-20",
    #                    'end_time': "2013-05-21"
    #                }           
    #            }
    #        },
    #    }
    #},
    #'test': {
    #    'datasets': {
    #        'UKDale': {
    #            'path': '../../../../datasets/ukdale/ukdale.h5',
    #            'buildings': {
    #                5: {
    #                    'start_time': "2014-07-01",
    #                    'end_time': "2014-07-02"
    #                } 
    #            }
    #        }
    #    },
    #    'metrics':['mae', 'rmse']
    #}
}

#### Training and testing Fridge ####
#results = API(fridge)
#
##Get all the results in the experiment and print them.
#errors_keys = results.errors_keys
#errors = results.errors
#
#for app in results.appliances:
#    f = open(base_path + "results_" + app.replace(" ", "_") + ".txt", "w")
#    for classifier in errors[0].columns:
#        f.write(5*"-" + classifier + "-"*5 + "\n")
#        for i in range(len(errors)):
#            f.write(errors_keys[i].split("_")[-1].upper() + " : " + str(errors[i][classifier][app]) + "\n")
#
#for m in results.methods:
#    if m not in ["co", "fhmm_exact", "hart85"]:
#        create_path(base_path + "models/" + m)
#        results.methods[m].save_model(base_path + "models/" + m)
#
###################################################
#
#### Training and testing Kettle ####
#results = API(kettle)
#
##Get all the results in the experiment and print them.
#errors_keys = results.errors_keys
#errors = results.errors
#
#for app in results.appliances:
#    f = open(base_path + "results_" + app.replace(" ", "_") + ".txt", "w")
#    for classifier in errors[0].columns:
#        f.write(5*"-" + classifier + "-"*5 + "\n")
#        for i in range(len(errors)):
#            f.write(errors_keys[i].split("_")[-1].upper() + " : " + str(errors[i][classifier][app]) + "\n")#
#
####################################################

### Training and testing others ####
results = API(others)

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