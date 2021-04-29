

from nilmtk.disaggregate import CO, FHMMExact, Mean, Hart85
from nilmtk.api import API

import warnings
warnings.filterwarnings("ignore")
import sys
import logging

#sys.stderr = open('./outputs/err.log', 'w')
#
#logging.basicConfig(level=logging.INFO,
#                    format='%(asctime)s %(levelname)s %(message)s',
#                    filename='./outputs/info.log',
#                    filemode='w')

import sys
sys.path.insert(1, "../../nilmtk-contrib")
sys.path.insert(1, "../../regression_models")

from dae import DAE
from seq2point import Seq2Point
from seq2seq import Seq2Seq
from afhmm_sac import AFHMM_SAC
from gru_dwt import GRU_DWT
from gacd import GACD



# Classifies Dish Washer, Fridge, Microwave
experiment1 = {
    'power': {'mains': ['apparent'],'appliance': ['active']},
    'sample_rate': 6,
    'appliances': ['fridge', 'microwave', 'dish washer'],
    'methods': {
        'DAE':DAE({'n_epochs':1,'batch_size':1024}),
        'Seq2Point':Seq2Point({'n_epochs':1,'batch_size':1024}),
        'Seq2Seq':Seq2Seq({'n_epochs':1,'batch_size':1024}),
        #'GRU_DWT':GRU_DWT( {
        #    "verbose" : 2,
        #    "training_results_path" : "/home/user/thesis_results/history/",
        #    "appliances" : {
        #        "fridge" : {
        #            "dwt_timewindow" : 12,
        #            "dwt_overlap" : 10,
        #            "examples_overlap" : 0,
        #            "examples_timewindow" : 300,
        #            "epochs" : 1,
        #            "batch_size" : 1024,
        #            "wavelet": 'bior2.2',
        #        },
        #        "microwave" : {
        #            "dwt_timewindow" : 12,
        #            "dwt_overlap" : 10,
        #            "examples_overlap" : 0,
        #            "examples_timewindow" : 300,
        #            "epochs" : 1,
        #            "batch_size" : 1024,
        #            "wavelet": 'bior2.2',
        #        },
        #        "dish washer" : {
        #            "dwt_timewindow" : 12,
        #            "dwt_overlap" : 10,
        #            "examples_overlap" : 0,
        #            "examples_timewindow" : 300,
        #            "epochs" : 1,
        #            "batch_size" : 1024,
        #            "wavelet": 'bior2.2',
        #        }
        #    },
        #    "predicted_column": ("power", "apparent"), 
        #}),
        #'GACD':GACD({
        #    "verbose" : 2,
        #    "training_results_path" : "/home/user/thesis_results/history/",
        #    "appliances" : {
        #        "fridge" : {
        #            "dwt_timewindow" : 12,
        #            "dwt_overlap" : 10,
        #            "examples_overlap" : 0,
        #            "examples_timewindow" : 300,
        #            "epochs" : 1,
        #            "batch_size" : 1024,
        #            "wavelet": 'bior2.2',
        #        },
        #        "microwave" : {
        #            "dwt_timewindow" : 12,
        #            "dwt_overlap" : 10,
        #            "examples_overlap" : 0,
        #            "examples_timewindow" : 300,
        #            "epochs" : 1,
        #            "batch_size" : 1024,
        #            "wavelet": 'bior2.2',
        #        },
        #        "dish washer" : {
        #            "dwt_timewindow" : 12,
        #            "dwt_overlap" : 10,
        #            "examples_overlap" : 0,
        #            "examples_timewindow" : 300,
        #            "epochs" : 1,
        #            "batch_size" : 1024,
        #            "wavelet": 'bior2.2',
        #        }
        #    },
        #    "predicted_column": ("power", "apparent"), 
        #})
    },
    'train': {    
        'datasets': {
            'UKDale': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    1: {
                        'start_time': "2013-05-22",
                        #'end_time':  "2013-07-01"
                        'end_time' : "2013-05-23"
                    },
                    2: {
                        'start_time': "2013-05-22",
                        #'end_time':  "2013-07-01"
                        'end_time' : "2013-05-23"
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
                        #'end_time': "2014-07-09"
                        'end_time': "2014-07-01"
                    }
                }
            }
        },
        'metrics':['mae', 'rmse']
    }
}

# Washing Machine
experiment2 = {
    'power': {'mains': ['apparent'],'appliance': ['active']},
    'sample_rate': 6,
    'appliances': ['washing machine'],
    'methods': {
        'DAE':DAE({'n_epochs':1,'batch_size':1024}),
        'Seq2Point':Seq2Point({'n_epochs':1,'batch_size':1024}),
        'Seq2Seq':Seq2Seq({'n_epochs':1,'batch_size':1024}),
        #'GRU_DWT':GRU_DWT( {
        #    "verbose" : 2,
        #    "training_results_path" : "/home/user/thesis_results/history/",
        #    "appliances" : {
        #        "washing machine" : {
        #            "dwt_timewindow" : 12,
        #            "dwt_overlap" : 10,
        #            "examples_overlap" : 0,
        #            "examples_timewindow" : 300,
        #            "epochs" : 1,
        #            "batch_size" : 1024,
        #            "wavelet": 'bior2.2',
        #        },
        #    },
        #    "predicted_column": ("power", "apparent"), 
        #}),
        #'GACD':GACD({
        #    "verbose" : 2,
        #    "training_results_path" : "/home/user/thesis_results/history/",
        #    "appliances" : {
        #        "washing machine" : {
        #            "dwt_timewindow" : 12,
        #            "dwt_overlap" : 10,
        #            "examples_overlap" : 0,
        #            "examples_timewindow" : 300,
        #            "epochs" : 1,
        #            "batch_size" : 1024,
        #            "wavelet": 'bior2.2',
        #        },
        #    },
        #    "predicted_column": ("power", "apparent"), 
        #})
    },
    'train': {    
        'datasets': {
            'UKDale': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    1: {
                        'start_time': "2013-05-22",
                        #'end_time':  "2013-07-01"
                        'end_time' : "2013-05-23"
                    }             
                }
            }
        }
    },
    'test': {
        'datasets': {
            'UKDale': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    2 : {
                        'start_time': "2013-05-22",
                        #'end_time':  "2013-06-01"
                        'end_time' : "2013-05-23"
                    }
                }
            }
        },
        'metrics':['mae', 'rmse']
    }
}

# Kettle

experiment3 = {
    'power': {'mains': ['apparent'],'appliance': ['active']},
    'sample_rate': 6,
    'appliances': ['kettle'],
    'methods': {
        'DAE':DAE({'n_epochs':1,'batch_size':1024}),
        'Seq2Point':Seq2Point({'n_epochs':1,'batch_size':1024}),
        'Seq2Seq':Seq2Seq({'n_epochs':1,'batch_size':1024}),
        #'GRU_DWT':GRU_DWT( {
        #    "verbose" : 2,
        #    "training_results_path" : "/home/user/thesis_results/history/",
        #    "appliances" : {
        #        "kettle" : {
        #            "dwt_timewindow" : 12,
        #            "dwt_overlap" : 10,
        #            "examples_overlap" : 0,
        #            "examples_timewindow" : 300,
        #            "epochs" : 1,
        #            "batch_size" : 1024,
        #            "wavelet": 'bior2.2',
        #        },
        #    },
        #    "predicted_column": ("power", "apparent"), 
        #}),
        #'GACD':GACD({
        #    "verbose" : 2,
        #    "training_results_path" : "/home/user/thesis_results/history/",
        #    "appliances" : {
        #        "kettle" : {
        #            "dwt_timewindow" : 12,
        #            "dwt_overlap" : 10,
        #            "examples_overlap" : 0,
        #            "examples_timewindow" : 300,
        #            "epochs" : 1,
        #            "batch_size" : 1024,
        #            "wavelet": 'bior2.2',
        #        },
        #    },
        #    "predicted_column": ("power", "apparent"), 
        #})
    },
    'train': {    
        'datasets': {
            'UKDale': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    2: {
                        'start_time': "2013-03-01",
                        #'end_time':  "2013-07-01"
                        'end_time' : "2013-03-02"
                    },
                    #3: {
                    #    'start_time': "2013-02-28",
                    #    #'end_time':  "2013-04-08"
                    #    'end_time' : "2013-02-29"
                    #}
                
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
                        #'end_time':  "2014-07-09"
                        'end_time' : "2014-07-01"
                    }
                }
            }
        },
        'metrics':['mae', 'rmse']
    }
}

# Toaster

experiment4 = {
    'power': {'mains': ['apparent'],'appliance': ['active']},
    'sample_rate': 6,
    'appliances': ['toaster'],
    'methods': {
        'DAE':DAE({'n_epochs':1,'batch_size':1024}),
        'Seq2Point':Seq2Point({'n_epochs':1,'batch_size':1024}),
        'Seq2Seq':Seq2Seq({'n_epochs':1,'batch_size':1024}),
        #'GRU_DWT':GRU_DWT( {
        #    "verbose" : 2,
        #    "training_results_path" : "/home/user/thesis_results/history/",
        #    "appliances" : {
        #        "toaster" : {
        #            "dwt_timewindow" : 12,
        #            "dwt_overlap" : 10,
        #            "examples_overlap" : 0,
        #            "examples_timewindow" : 300,
        #            "epochs" : 1,
        #            "batch_size" : 1024,
        #            "wavelet": 'bior2.2',
        #        },
        #    },
        #    "predicted_column": ("power", "apparent"), 
        #}),
        #'GACD':GACD({
        #    "verbose" : 2,
        #    "training_results_path" : "/home/user/thesis_results/history/",
        #    "appliances" : {
        #        "toaster" : {
        #            "dwt_timewindow" : 12,
        #            "dwt_overlap" : 10,
        #            "examples_overlap" : 0,
        #            "examples_timewindow" : 300,
        #            "epochs" : 1,
        #            "batch_size" : 1024,
        #            "wavelet": 'bior2.2',
        #        },
        #    },
        #    "predicted_column": ("power", "apparent"), 
        #})
    },
    'train': {    
        'datasets': {
            'UKDale': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    2: {
                        'start_time': "2013-05-22",
                        #'end_time':  "2013-07-01"
                        'end_time' : "2013-05-23"
                    }
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
                        'start_time': "2014-06-29",
                        #'end_time':  "2014-07-09"
                        'end_time' : "2014-06-30"
                    }
                }
            }
        },
        'metrics':['mae', 'rmse']
    }
}

# Oven

experiment5 = {
    'power': {'mains': ['apparent'],'appliance': ['active']},
    'sample_rate': 6,
    'appliances': ['oven'],
    'methods': {
        'DAE':DAE({'n_epochs':1,'batch_size':1024}),
        'Seq2Point':Seq2Point({'n_epochs':1,'batch_size':1024}),
        'Seq2Seq':Seq2Seq({'n_epochs':1,'batch_size':1024}),
        #'GRU_DWT':GRU_DWT( {
        #    "verbose" : 2,
        #    "training_results_path" : "/home/user/thesis_results/history/",
        #    "appliances" : {
        #        "oven" : {
        #            "dwt_timewindow" : 12,
        #            "dwt_overlap" : 10,
        #            "examples_overlap" : 0,
        #            "examples_timewindow" : 300,
        #            "epochs" : 1,
        #            "batch_size" : 1024,
        #            "wavelet": 'bior2.2',
        #        },
        #    },
        #    "predicted_column": ("power", "apparent"), 
        #}),
        #'GACD':GACD({
        #    "verbose" : 2,
        #    "training_results_path" : "/home/user/thesis_results/history/",
        #    "appliances" : {
        #        "oven" : {
        #            "dwt_timewindow" : 12,
        #            "dwt_overlap" : 10,
        #            "examples_overlap" : 0,
        #            "examples_timewindow" : 300,
        #            "epochs" : 1,
        #            "batch_size" : 1024,
        #            "wavelet": 'bior2.2',
        #        },
        #    },
        #    "predicted_column": ("power", "apparent"), 
        #})
    },
    'train': {    
        'datasets': {
            'UKDale': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    5: {
                        'start_time': "2014-06-29",
                        #'end_time':  "2014-11-01"
                        'end_time':  "2014-06-30"
                    }
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
                        'start_time': "2014-11-01",
                        #'end_time': None
                        'end_time':  "2014-11-02"
                    }
                }   
            }
        },
        'metrics':['mae', 'rmse']
    }
}

# Boiler

experiment6 = {
    'power': {'mains': ['apparent'],'appliance': ['apparent']},
    'sample_rate': 6,
    'appliances': ['boiler'],
    'methods': {
        'DAE':DAE({'n_epochs':1,'batch_size':1024}),
        'Seq2Point':Seq2Point({'n_epochs':1,'batch_size':1024}),
        'Seq2Seq':Seq2Seq({'n_epochs':1,'batch_size':1024}),
        #'GRU_DWT':GRU_DWT( {
        #    "verbose" : 2,
        #    "training_results_path" : "/home/user/thesis_results/history/",
        #    "appliances" : {
        #        "boiler" : {
        #            "dwt_timewindow" : 12,
        #            "dwt_overlap" : 10,
        #            "examples_overlap" : 0,
        #            "examples_timewindow" : 300,
        #            "epochs" : 1,
        #            "batch_size" : 1024,
        #            "wavelet": 'bior2.2',
        #        },
        #    },
        #    "predicted_column": ("power", "apparent"), 
        #}),
        #'GACD':GACD({
        #    "verbose" : 2,
        #    "training_results_path" : "/home/user/thesis_results/history/",
        #    "appliances" : {
        #        "boiler" : {
        #            "dwt_timewindow" : 12,
        #            "dwt_overlap" : 10,
        #            "examples_overlap" : 0,
        #            "examples_timewindow" : 300,
        #            "epochs" : 1,
        #            "batch_size" : 1024,
        #            "wavelet": 'bior2.2',
        #        },
        #    },
        #    "predicted_column": ("power", "apparent"), 
        #})
    },
    'train': {    
        'datasets': {
            'UKDale': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    1: {
                        'start_time': "2013-05-22",
                        #'end_time':  "2013-07-01"
                        'end_time' : "2013-05-23"
                    }
                }
            }
        }
    },
    'test': {
        'datasets': {
            'UKDale': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    1: {
                        'start_time': "2017-01-01",
                        #'end_time':  "2017-01-09"
                        'end_time' : "2017-01-02"
                    }
                }
            }
        },
        'metrics':['mae', 'rmse']
    }
}

models_folder = "./models/"

api_results_experiment_1 = API(experiment1)


#Get all the results in the experiment and print them.
errors_keys = api_results_experiment_1.errors_keys
errors = api_results_experiment_1.errors
for i in range(len(errors)):
    print (errors_keys[i])
    print (errors[i])
    print ("\n\n")

api_results_experiment_2 = API(experiment2)


#Get all the results in the experiment and print them.
errors_keys = api_results_experiment_2.errors_keys
errors = api_results_experiment_2.errors
for i in range(len(errors)):
    print (errors_keys[i])
    print (errors[i])
    print ("\n\n")

#api_results_experiment_3 = API(experiment3)
#
#
##Get all the results in the experiment and print them.
#errors_keys = api_results_experiment_3.errors_keys
#errors = api_results_experiment_3.errors
#for i in range(len(errors)):
#    print (errors_keys[i])
#    print (errors[i])
#    print ("\n\n")

api_results_experiment_4 = API(experiment4)


#Get all the results in the experiment and print them.
errors_keys = api_results_experiment_4.errors_keys
errors = api_results_experiment_4.errors
for i in range(len(errors)):
    print (errors_keys[i])
    print (errors[i])
    print ("\n\n")


#api_results_experiment_5 = API(experiment5)
#
##Get all the results in the experiment and print them.
#errors_keys = api_results_experiment_5.errors_keys
#errors = api_results_experiment_5.errors
#for i in range(len(errors)):
#    print (errors_keys[i])
#    print (errors[i])
#    print ("\n\n")

api_results_experiment_6 = API(experiment6)


#Get all the results in the experiment and print them.
errors_keys = api_results_experiment_6.errors_keys
errors = api_results_experiment_6.errors
for i in range(len(errors)):
    print (errors_keys[i])
    print (errors[i])
    print ("\n\n")


