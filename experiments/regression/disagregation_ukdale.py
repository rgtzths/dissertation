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
epochs = 500

# Classifies Dish Washer, Fridge, Microwave
fridge_freezer = {
    'power': {'mains': ['apparent'],'appliance': ['active']},
    'sample_rate': 6,
    'appliances': ['fridge freezer'],
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
                "fridge freezer" : {
                    'timewindow' : 180,
                    'timestep' : 2,
                    'overlap' : 178,
                    'epochs' : epochs,
                    'batch_size' : 1024,
                    'n_nodes' : 90
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
                "fridge freezer" : {
                    'timewindow' : 180,
                    'timestep' : 2,
                    'overlap' : 178,
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
                "fridge freezer" : {
                    'timewindow' : 180,
                    'timestep' : 2,
                    'overlap' : 178,
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
            'UKDale': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    1: {
                        'start_time': "2014-02-01",
                        'end_time' : "2014-03-01"
                    },             
                }
            },
            'UKDale2': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    1: {
                        'start_time': "2016-09-01",
                        'end_time' : "2016-10-01"
                    },             
                }
            }
        }
    },
    'test': {
        'datasets': {
            'UKDale': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    5: {
                        'start_time': "2014-06-30",
                        'end_time': "2014-07-30"
                    }
                }
            }
        },
        'metrics':['mae', 'rmse']
    }
    #'train': {    
    #    'datasets': {
    #        'UKDale': {
    #            'path': '../../../datasets/ukdale/ukdale.h5',
    #            'buildings': {
    #                1: {
    #                    'start_time': "2014-02-01",
    #                    'end_time' : "2014-02-02"
    #                },             
    #            }
    #        }
    #    }
    #},
    #'test': {
    #    'datasets': {
    #        'UKDale': {
    #            'path': '../../../datasets/ukdale/ukdale.h5',
    #            'buildings': {
    #                5: {
    #                    'start_time': "2014-06-29",
    #                    'end_time': "2014-06-30"
    #                }
    #            }
    #        }
    #    },
    #    'metrics':['mae', 'rmse']
    #}
}

microwave = {
    'power': {'mains': ['apparent'],'appliance': ['active']},
    'sample_rate': 6,
    'appliances': ['microwave'],
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
                "microwave" : {
                    'timewindow' : 180,
                    'timestep' : 2,
                    'overlap' : 178,
                    'epochs' : epochs,
                    'batch_size' : 1024,
                    'n_nodes' : 90
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
                    'timewindow' : 180,
                    'timestep' : 2,
                    'overlap' : 178,
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
                "microwave" : {
                    'timewindow' : 180,
                    'timestep' : 2,
                    'overlap' : 178,
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
            'UKDale': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    1: {
                        'start_time': "2013-06-25",
                        'end_time' : "2013-07-09"
                    },
                    5: {
                        'start_time': "2014-07-08",
                        'end_time' : "2014-07-09"
                    },             
                }
            },
            'UKDale2': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    1: {
                        'start_time': "2013-07-12",
                        'end_time' : "2013-07-18"
                    },
                    5: {
                        'start_time': "2014-07-10",
                        'end_time' : "2014-07-11"
                    },            
                }
            },
            'UKDale3': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    1: {
                        'start_time': "2013-07-28",
                        'end_time' : "2013-08-09"
                    },
                    5: {
                        'start_time': "2014-07-11",
                        'end_time' : "2014-07-17"
                    },          
                }
            },
            'UKDale4': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    5: {
                        'start_time': "2014-07-21",
                        'end_time' : "2014-07-22"
                    },             
                }
            },
            'UKDale5': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    5: {
                        'start_time': "2014-07-28",
                        'end_time' : "2014-07-29"
                    },           
                }
            },
            'UKDale6': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    5: {
                        'start_time': "2014-08-04",
                        'end_time' : "2014-08-05"
                    },            
                }
            },
            'UKDale7': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    5: {
                        'start_time': "2014-08-07",
                        'end_time' : "2014-08-08"
                    },             
                }
            },
            'UKDale8': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    5: {
                        'start_time': "2014-08-10",
                        'end_time' : "2014-08-12"
                    },           
                }
            },
            'UKDale9': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {            
                    5: {
                        'start_time': "2014-08-13",
                        'end_time' : "2014-08-16"
                    }, 
                }
            },
            'UKDale10': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    5: {
                        'start_time': "2014-08-20",
                        'end_time' : "2014-08-25"
                    },            
                }
            },
            'UKDale11': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    5: {
                        'start_time': "2014-08-26",
                        'end_time' : "2014-08-28"
                    },            
                }
            },
            'UKDale12': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    5: {
                        'start_time': "2014-08-29",
                        'end_time' : "2014-09-02"
                    },             
                }
            },
            'UKDale13': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    5: {
                        'start_time': "2014-09-05",
                        'end_time' : "2014-09-06"
                    },             
                }
            },
            'UKDale14': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    5: {
                        'start_time': "2014-09-07",
                        'end_time' : "2014-09-08"
                    },           
                }
            },
        }
    },
    'test': {
        'datasets': {
            'UKDale': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    2: {
                        'start_time': "2013-06-06",
                        'end_time': "2013-07-05"
                    }
                }
            },
        },
        'metrics':['mae', 'rmse']
    }
    #'train': {    
    #    'datasets': {
    #        'UKDale': {
    #            'path': '../../../datasets/ukdale/ukdale.h5',
    #            'buildings': {
    #                1: {
    #                    'start_time': "2014-02-01",
    #                    'end_time' : "2014-02-02"
    #                },             
    #            }
    #        }
    #    }
    #},
    #'test': {
    #    'datasets': {
    #        'UKDale': {
    #            'path': '../../../datasets/ukdale/ukdale.h5',
    #            'buildings': {
    #                5: {
    #                    'start_time': "2014-06-29",
    #                    'end_time': "2014-06-30"
    #                }
    #            }
    #        }
    #    },
    #    'metrics':['mae', 'rmse']
    #}
}

dish_washer = {
    'power': {'mains': ['apparent'],'appliance': ['active']},
    'sample_rate': 6,
    'appliances': ['dish washer'],
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
                "dish washer" : {
                    'timewindow' : 180,
                    'timestep' : 2,
                    'overlap' : 178,
                    'epochs' : epochs,
                    'batch_size' : 1024,
                    'n_nodes' : 90
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
                    'timewindow' : 180,
                    'timestep' : 2,
                    'overlap' : 178,
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
                    'timewindow' : 180,
                    'timestep' : 2,
                    'overlap' : 178,
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
            'UKDale': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    1: {
                        'start_time': "2013-02-18",
                        'end_time' : "2013-02-19"
                    },
                    5: {
                        'start_time': "2014-06-30",
                        'end_time' : "2014-07-01"
                    },             
                }
            },
            'UKDale2': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    1: {
                        'start_time': "2013-02-21",
                        'end_time' : "2013-02-22"
                    },
                    5: {
                        'start_time': "2014-07-03",
                        'end_time' : "2014-07-04"
                    },            
                }
            },
            'UKDale3': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    1: {
                        'start_time': "2013-02-24",
                        'end_time' : "2013-02-25"
                    },
                    5: {
                        'start_time': "2014-07-06",
                        'end_time' : "2014-07-07"
                    },          
                }
            },
            'UKDale4': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    1: {
                        'start_time': "2013-02-27",
                        'end_time' : "2013-02-28"
                    },
                    5: {
                        'start_time': "2014-07-08",
                        'end_time' : "2014-07-09"
                    },             
                }
            },
            'UKDale5': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    1: {
                        'start_time': "2013-03-09",
                        'end_time' : "2013-02-10"
                    },
                    5: {
                        'start_time': "2014-07-11",
                        'end_time' : "2014-07-12"
                    },           
                }
            },
            'UKDale6': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    1: {
                        'start_time': "2013-03-16",
                        'end_time' : "2013-03-17"
                    },
                    5: {
                        'start_time': "2014-07-13",
                        'end_time' : "2014-07-14"
                    },            
                }
            },
            'UKDale7': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    1: {
                        'start_time': "2013-03-19",
                        'end_time' : "2013-03-20"
                    },
                    5: {
                        'start_time': "2014-07-19",
                        'end_time' : "2014-07-20"
                    },             
                }
            },
            'UKDale8': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    1: {
                        'start_time': "2013-03-21",
                        'end_time' : "2013-03-22"
                    },
                    5: {
                        'start_time': "2014-07-22",
                        'end_time' : "2014-07-23"
                    },           
                }
            },
            'UKDale9': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    1: {
                        'start_time': "2013-04-02",
                        'end_time' : "2013-04-03"
                    },             
                    5: {
                        'start_time': "2014-07-27",
                        'end_time' : "2014-07-28"
                    }, 
                }
            },
            'UKDale10': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    1: {
                        'start_time': "2013-04-08",
                        'end_time' : "2013-04-09"
                    },
                    5: {
                        'start_time': "2014-07-29",
                        'end_time' : "2014-07-30"
                    },            
                }
            },
            'UKDale11': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    1: {
                        'start_time': "2013-04-26",
                        'end_time' : "2013-04-27"
                    },
                    5: {
                        'start_time': "2014-08-01T12:00",
                        'end_time' : "2014-08-02T12:00"
                    },            
                }
            },
            'UKDale12': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    1: {
                        'start_time': "2013-04-28T12:00",
                        'end_time' : "2013-04-29T12:00"
                    },
                    5: {
                        'start_time': "2014-08-03",
                        'end_time' : "2014-08-04"
                    },             
                }
            },
            'UKDale13': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    1: {
                        'start_time': "2013-05-01T12:00",
                        'end_time' : "2013-05-02T12:00"
                    },
                    5: {
                        'start_time': "2014-08-08",
                        'end_time' : "2014-08-09"
                    },             
                }
            },
            'UKDale14': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    1: {
                        'start_time': "2013-05-04T12:00",
                        'end_time' : "2013-05-05T12:00"
                    },
                    5: {
                        'start_time': "2014-08-10",
                        'end_time' : "2014-08-11"
                    },           
                }
            },
            'UKDale15': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    1: {
                        'start_time': "2013-05-10",
                        'end_time' : "2013-05-11"
                    },
                    5: {
                        'start_time': "2014-08-13",
                        'end_time' : "2014-08-14"
                    },             
                }
            },
            'UKDale16': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    1: {
                        'start_time': "2013-05-14T12:00",
                        'end_time' : "2013-05-15T12:00"
                    },
                    5: {
                        'start_time': "2014-08-16",
                        'end_time' : "2014-08-17"
                    },             
                }
            },
            'UKDale17': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    1: {
                        'start_time': "2013-05-18",
                        'end_time' : "2013-05-19"
                    },
                    5: {
                        'start_time': "2014-08-19",
                        'end_time' : "2014-08-20"
                    },     
                }
            },
            'UKDale18': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    1: {
                        'start_time': "2013-05-20T12:00",
                        'end_time' : "2013-05-21T12:00"
                    },
                    5: {
                        'start_time': "2014-08-22",
                        'end_time' : "2014-08-23"
                    },             
                }
            },
            'UKDale19': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    1: {
                        'start_time': "2013-05-22T12:00",
                        'end_time' : "2013-05-23T12:00"
                    },
                    5: {
                        'start_time': "2014-08-24",
                        'end_time' : "2014-08-25"
                    },               
                }
            },
            'UKDale20': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    1: {
                        'start_time': "2013-05-31",
                        'end_time' : "2013-06-1"
                    },
                    5: {
                        'start_time': "2014-08-27",
                        'end_time' : "2014-08-28"
                    },           
                }
            },
            'UKDale21': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    1: {
                        'start_time': "2013-06-02",
                        'end_time' : "2013-06-03"
                    },
                    5: {
                        'start_time': "2014-08-29",
                        'end_time' : "2014-08-31"
                    },             
                }
            },
            'UKDale22': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    1: {
                        'start_time': "2013-06-04",
                        'end_time' : "2013-06-07"
                    },             
                    5: {
                        'start_time': "2014-09-01",
                        'end_time' : "2014-09-02"
                    },  
                }
            },
            'UKDale23': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    1: {
                        'start_time': "2013-06-08",
                        'end_time' : "2013-06-09"
                    },
                    5: {
                        'start_time': "2014-09-03",
                        'end_time' : "2014-09-04"
                    },             
                }
            },
            'UKDale24': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    1: {
                        'start_time': "2013-06-10",
                        'end_time' : "2013-06-12"
                    },
                    5: {
                        'start_time': "2014-09-06",
                        'end_time' : "2014-09-07"
                    },             
                }
            },
            'UKDale25': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    1: {
                        'start_time': "2013-06-19",
                        'end_time' : "2013-06-20"
                    },
                    5: {
                        'start_time': "2014-09-09",
                        'end_time' : "2014-09-10"
                    },             
                }
            },
            'UKDale26': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    1: {
                        'start_time': "2013-06-23",
                        'end_time' : "2013-06-24"
                    },
                    5: {
                        'start_time': "2014-09-14",
                        'end_time' : "2014-09-16"
                    },             
                }
            },
            'UKDale27': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    1: {
                        'start_time': "2013-06-28",
                        'end_time' : "2013-06-29"
                    },
                    5: {
                        'start_time': "2014-09-17",
                        'end_time' : "2014-09-18"
                    },            
                }
            },
            'UKDale28': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    5: {
                        'start_time': "2014-09-19",
                        'end_time' : "2014-09-20"
                    },            
                }
            },
        }
    },
    'test': {
        'datasets': {
            'UKDale': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    2: {
                        'start_time': "2013-05-26",
                        'end_time': "2013-06-23"
                    }
                }
            },
        },
        'metrics':['mae', 'rmse']
    }
    #'train': {    
    #    'datasets': {
    #        'UKDale': {
    #            'path': '../../../datasets/ukdale/ukdale.h5',
    #            'buildings': {
    #                1: {
    #                    'start_time': "2014-02-01",
    #                    'end_time' : "2014-02-02"
    #                },             
    #            }
    #        }
    #    }
    #},
    #'test': {
    #    'datasets': {
    #        'UKDale': {
    #            'path': '../../../datasets/ukdale/ukdale.h5',
    #            'buildings': {
    #                5: {
    #                    'start_time': "2014-06-29",
    #                    'end_time': "2014-06-30"
    #                }
    #            }
    #        }
    #    },
    #    'metrics':['mae', 'rmse']
    #}
}

### Training and testing Fridge ####
results = API(fridge_freezer)

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

