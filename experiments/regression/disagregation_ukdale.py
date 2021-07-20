from nilmtk.api import API

import sys
sys.path.insert(1, "../../nilmtk-contrib")
sys.path.insert(1, "../../regression_models")
sys.path.insert(1, "../../utils")
sys.path.insert(1, "../../feature_extractors")

from dae import DAE
from seq2point import Seq2Point
from seq2seq import Seq2Seq
from rnn import RNN
from WindowGRU import WindowGRU

from resnet import ResNet
from deep_gru import DeepGRU
from mlp_dwt import MLP

base_path= "/home/rteixeira/soa_results/"
#base_path = "/home/user/article_results/"
#base_path = "/home/atnoguser/article_results/"
epochs = 500
timestep = 6
timewindow = 64 * timestep
overlap = timewindow - timestep

# Classifies Dish Washer, Fridge, Microwave, washing machine and kettle
fridge = {
    'power': {'mains': ['apparent'],'appliance': ['active']},
    'sample_rate': timestep,
    'appliances': ['fridge'],
    'methods': {
        'DAE':DAE({
            'n_epochs':epochs,
            'sequence_length':299,
            "training_history_folder" : base_path + "history/DAE/",
            "plots_folder" : base_path + "plots/DAE/",
            "file_prefix" : base_path + "models/DAE/",
            "mains_mean" : 594.9043,
            "mains_std" : 513.311,
            "on_threshold" : 50
        }),
        'Seq2Point':Seq2Point({
            'n_epochs':epochs,
            'sequence_length':299,
            "training_history_folder" : base_path + "history/Seq2Point/",
            "plots_folder" : base_path + "plots/Seq2Point/",
            "file_prefix" : base_path + "models/Seq2Point/",
            "mains_mean" : 594.9043,
            "mains_std" : 513.311,
            "on_threshold" : 50
        }),
        'Seq2Seq':Seq2Seq({
            'n_epochs':epochs,
            "batch_size" : 256,
            'sequence_length':299,
            "training_history_folder" : base_path + "history/Seq2Seq/",
            "plots_folder" : base_path + "plots/Seq2Seq/",
            "file_prefix" : base_path + "models/Seq2Seq/",
            "mains_mean" : 594.9043,
            "mains_std" : 513.311,
            "on_threshold" : 50
        }),
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
            "checkpoint_folder" : base_path + "models/DeepGRU/",
            "plots_folder" : base_path + "plots/DeepGRU/",
            #"load_model_folder" : base_path + "models/DeepGRU/",
            "appliances" : {
                "fridge" : {
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
            "checkpoint_folder" : base_path + "models/MLP/",
            "plots_folder" : base_path + "plots/MLP/",
            #"load_model_folder" : base_path + "models/MLP/",
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
            "checkpoint_folder" : base_path + "models/MLP_Raw/",
            "plots_folder" : base_path + "plots/MLP_Raw/",
            #"load_model_folder" : base_path + "models/MLP_Raw/",
            "appliances" : {
                "fridge" : {
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
            #'Eco': {
            #    'path': '../../../../datasets/eco_h5/eco.h5',
            #    'buildings': {
            #        2: {
            #            'start_time': "2012-07-01",
            #            'end_time': "2012-08-01"
            #        } 
            #    }
            #}
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
    #                    'start_time': "2013-06-30",
    #                    'end_time': "2013-07-01"
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
    #                    'start_time': "2013-05-22",
    #                    'end_time': "2013-05-23"
    #                } 
    #            }
    #        }
    #    },
    #    'metrics':['mae', 'rmse']
    #}
}

microwave = {
    'power': {'mains': ['apparent'],'appliance': ['active']},
    'sample_rate': timestep,
    'appliances': ['microwave'],
    'methods': {
        'DAE':DAE({
            'n_epochs':epochs,
            'sequence_length':299,
            "training_history_folder" : base_path + "history/DAE/",
            "plots_folder" : base_path + "plots/DAE/",
            "file_prefix" : base_path + "models/DAE/",
            "mains_mean" : 594.9043,
            "mains_std" : 513.311,
            "on_threshold" : 200
        }),
        'Seq2Point':Seq2Point({
            'n_epochs':epochs,
            'sequence_length':299,
            "training_history_folder" : base_path + "history/Seq2Point/",
            "plots_folder" : base_path + "plots/Seq2Point/",
            "file_prefix" : base_path + "models/Seq2Point/",
            "mains_mean" : 594.9043,
            "mains_std" : 513.311,
            "on_threshold" : 200
        }),
        'Seq2Seq':Seq2Seq({
            'n_epochs':epochs,
            'sequence_length':299,
            "batch_size" : 256,
            "training_history_folder" : base_path + "history/Seq2Seq/",
            "plots_folder" : base_path + "plots/Seq2Seq/",
            "file_prefix" : base_path + "models/Seq2Seq/",
            "mains_mean" : 594.9043,
            "mains_std" : 513.311,
            "on_threshold" : 200
        }),
        "ResNet" : ResNet( {
            "verbose" : 2,
            "training_history_folder" : base_path + "history/ResNet/",
            "results_folder" : base_path + "results/ResNet/",
            "checkpoint_folder" : base_path + "models/ResNet/",
            "plots_folder" : base_path + "plots/ResNet/",
            #"load_model_folder" : base_path + "models/ResNet/",
            "appliances" : {
                "microwave" : {
                    'timewindow' : timewindow,
                    'timestep' : timestep,
                    'overlap' :  overlap,
                    'epochs' : epochs,
                    'batch_size' : 256,
                    'n_nodes' : 32,
                    'on_treshold' : 200
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
                "microwave" : {
                    'timewindow' : timewindow,
                    'timestep' : timestep,
                    'overlap' :  overlap,
                    'epochs' : epochs,
                    'batch_size' : 256,
                    'n_nodes' : 90,
                    'on_treshold' : 200
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
                "microwave" : {
                    'timewindow' : timewindow,
                    'timestep' : timestep,
                    'overlap' :  overlap,
                    'epochs' : epochs,
                    'batch_size' : 256,
                    'feature_extractor' : "wt",
                    'on_treshold' : 200
                }
            },
            "predicted_column": ("power", "active"), 
        }),
        "MLP_Raw" : MLP( {
            "verbose" : 2,
            "training_history_folder" : base_path + "history/MLP_Raw/",
            "results_folder" : base_path + "results/MLP_Raw/",
            "checkpoint_folder" : base_path + "models/MLP_Raw/",
            "plots_folder" : base_path + "plots/MLP_Raw/",
            #"load_model_folder" : base_path + "models/MLP_Raw/",
            "appliances" : {
                "microwave" : {
                    'timewindow' : timewindow,
                    'timestep' : timestep,
                    'overlap' :  overlap,
                    'epochs' : epochs,
                    'batch_size' : 256,
                    'on_treshold' : 200
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
                    1: {
                        'start_time': "2013-04-17",
                        'end_time': "2013-10-09",
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
            'Eco': {
                'path': '../../../../datasets/eco_h5/eco.h5',
                'buildings': {
                    4: {
                        'start_time': "2012-12-01",
                        'end_time': "2013-01-01"
                    } 
                }
            }
        },
        'metrics':['mae', 'rmse']
    }

}

dish_washer = {
    'power': {'mains': ['apparent'],'appliance': ['active']},
    'sample_rate': timestep,
    'appliances': ['dish washer'],
    'methods': {
        'DAE':DAE({
            'n_epochs':epochs,
            'sequence_length':299,
            "training_history_folder" : base_path + "history/DAE/",
            "plots_folder" : base_path + "plots/DAE/",
            "file_prefix" : base_path + "models/DAE/",
            "mains_mean" : 613.3364,
            "mains_std" : 612.0515,
            "on_threshold" : 50
        }),
        'Seq2Point':Seq2Point({
            'n_epochs':epochs,
            'sequence_length':299,
            "training_history_folder" : base_path + "history/Seq2Point/",
            "plots_folder" : base_path + "plots/Seq2Point/",
            "file_prefix" : base_path + "models/Seq2Point/",
            "mains_mean" : 613.3364,
            "mains_std" : 612.0515,
            "on_threshold" : 50
        }),
        'Seq2Seq':Seq2Seq({
            'n_epochs':epochs,
            'sequence_length':299,
            "batch_size" : 256,
            "training_history_folder" : base_path + "history/Seq2Seq/",
            "plots_folder" : base_path + "plots/Seq2Seq/",
            "file_prefix" : base_path + "models/Seq2Seq/",
            "mains_mean" : 613.3364,
            "mains_std" : 612.0515,
            "on_threshold" : 50
        }),
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
                    'overlap' :  overlap,
                    'epochs' : epochs,
                    'batch_size' : 256,
                    'n_nodes' : 32,
                    "on_threshold" : 50
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
            "checkpoint_folder" : base_path + "models/MLP/",
            "plots_folder" : base_path + "plots/MLP/",
            #"load_model_folder" : base_path + "models/MLP/",
            "appliances" : {
                "dish washer" : {
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
            "checkpoint_folder" : base_path + "models/MLP_Raw/",
            "plots_folder" : base_path + "plots/MLP_Raw/",
            #"load_model_folder" : base_path + "models/MLP_Raw/",
            "appliances" : {
                "dish washer" : {
                    'timewindow' : timewindow,
                    'timestep' : timestep,
                    'overlap' :  overlap,
                    'epochs' : epochs,
                    'batch_size' : 256,
                    'on_treshold' : 50
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
                    1: {
                        'start_time': "2013-04-17",
                        'end_time': "2013-10-09",
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
            #'Eco': {
            #    'path': '../../../../datasets/eco_h5/eco.h5',
            #    'buildings': {
            #        2: {
            #            'start_time': "2012-10-01",
            #            'end_time': "2012-11-01"
            #        } 
            #    }
            #}
        },
        'metrics':['mae', 'rmse']
    }
}

kettle = {
    'power': {'mains': ['apparent'],'appliance': ['active']},
    'sample_rate': timestep,
    'appliances': ['kettle'],
    'methods': {
        'DAE':DAE({
            'n_epochs':epochs,
            'sequence_length':299,
            "training_history_folder" : base_path + "history/DAE/",
            "plots_folder" : base_path + "plots/DAE/",
            "file_prefix" : base_path + "models/DAE/",
            "mains_mean" : 685.76843,
            "mains_std" : 644.20844,
            "on_threshold" : 2000
        }),
        'Seq2Point':Seq2Point({
            'n_epochs':epochs,
            'sequence_length':299,
            "training_history_folder" : base_path + "history/Seq2Point/",
            "plots_folder" : base_path + "plots/Seq2Point/",
            "file_prefix" : base_path + "models/Seq2Point/",
            "mains_mean" : 685.76843,
            "mains_std" : 644.20844,
            "on_threshold" : 2000
        }),
        'Seq2Seq':Seq2Seq({
            'n_epochs':epochs,
            'sequence_length':299,
            "batch_size" : 256,
            "training_history_folder" : base_path + "history/Seq2Seq/",
            "plots_folder" : base_path + "plots/Seq2Seq/",
            "file_prefix" : base_path + "models/Seq2Seq/",
            "mains_mean" : 685.76843,
            "mains_std" : 644.20844,
            "on_threshold" : 2000
        }),
        "ResNet" : ResNet( {
            "verbose" : 2,
            "training_history_folder" : base_path + "history/ResNet/",
            "results_folder" : base_path + "results/ResNet/",
            "checkpoint_folder" : base_path + "models/ResNet/",
            "plots_folder" : base_path + "plots/ResNet/",
            #"load_model_folder" : base_path + "models/ResNet/",
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
            "checkpoint_folder" : base_path + "models/DeepGRU/",
            "plots_folder" : base_path + "plots/DeepGRU/",
            #"load_model_folder" : base_path + "models/DeepGRU/",
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
            "checkpoint_folder" : base_path + "models/MLP/",
            "plots_folder" : base_path + "plots/MLP/",
            #"load_model_folder" : base_path + "models/MLP/",
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
            "checkpoint_folder" : base_path + "models/MLP_Raw/",
            "plots_folder" : base_path + "plots/MLP_Raw/",
            #"load_model_folder" : base_path + "models/MLP_Raw/",
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
            #'Eco': {
            #    'path': '../../../../datasets/eco_h5/eco.h5',
            #    'buildings': {
            #        2: {
            #            'start_time': "2012-12-01",
            #            'end_time': "2013-01-01"
            #        } 
            #    }
            #}
        },
        'metrics':['mae', 'rmse']
    }

}

washing_machine = {
    'power': {'mains': ['apparent'],'appliance': ['active']},
    'sample_rate': timestep,
    'appliances': ['washing machine'],
    'methods': {
        'DAE':DAE({
            'n_epochs':epochs,
            'sequence_length':299,
            "training_history_folder" : base_path + "history/DAE/",
            "plots_folder" : base_path + "plots/DAE/",
            "file_prefix" : base_path + "models/DAE/",
            "mains_mean" : 633.60345,
            "mains_std" : 574.15015,
            "on_threshold" : 50
        }),
        'Seq2Point':Seq2Point({
            'n_epochs':epochs,
            'sequence_length':299,
            "training_history_folder" : base_path + "history/Seq2Point/",
            "plots_folder" : base_path + "plots/Seq2Point/",
            "file_prefix" : base_path + "models/Seq2Point/",
            "mains_mean" : 633.60345,
            "mains_std" : 574.15015,
            "on_threshold" : 50
        }),
        'Seq2Seq':Seq2Seq({
            'n_epochs':epochs,
            'sequence_length':299,
            "batch_size" : 256,
            "training_history_folder" : base_path + "history/Seq2Seq/",
            "plots_folder" : base_path + "plots/Seq2Seq/",
            "file_prefix" : base_path + "models/Seq2Seq/",
            "mains_mean" : 633.60345,
            "mains_std" : 574.15015,
            "on_threshold" : 50
        }),
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
            "checkpoint_folder" : base_path + "models/DeepGRU/",
            "plots_folder" : base_path + "plots/DeepGRU/",
            #"load_model_folder" : base_path + "models/DeepGRU/",
            "appliances" : {
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
            "checkpoint_folder" : base_path + "models/MLP/",
            "plots_folder" : base_path + "plots/MLP/",
            #"load_model_folder" : base_path + "models/MLP/",
            "appliances" : {
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
            "checkpoint_folder" : base_path + "models/MLP_Raw/",
            "plots_folder" : base_path + "plots/MLP_Raw/",
            #"load_model_folder" : base_path + "models/MLP_Raw/",
            "appliances" : {
                "washing machine" : {
                    'timewindow' : timewindow,
                    'timestep' : timestep,
                    'overlap' :  overlap,
                    'epochs' : epochs,
                    'batch_size' : 256,
                    'on_treshold' : 50
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
                    1: {
                        'start_time': "2013-04-17",
                        'end_time': "2013-10-09",
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
            #'Eco': {
            #    'path': '../../../../datasets/eco_h5/eco.h5',
            #    'buildings': {
            #        1: {
            #            'start_time': "2012-12-01",
            #            'end_time': "2013-01-01"
            #        } 
            #    }
            #}
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

### Training and testing Microwave ####
results = API(microwave)

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


### Training and testing Kettle ####
results = API(kettle)

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