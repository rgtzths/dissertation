
from nilmtk.api import API

import sys
sys.path.insert(1, "../../nilmtk-contrib")
sys.path.insert(1, "../../regression_models")

from dae import DAE
from seq2point import Seq2Point
from seq2seq import Seq2Seq
from afhmm_sac import AFHMM_SAC
from gru_dwt import GRU_DWT
from gacd import GACD

import warnings
warnings.filterwarnings("ignore")

#Experiment Definition
experiment1 = {
    'power': {'mains': ['apparent'], 'appliance': ['apparent']},
    'sample_rate': 2,
    'appliances': ['charger'],
    'methods': {
        'DAE':DAE({'n_epochs':1,'batch_size':1024}),
        'Seq2Point':Seq2Point({'n_epochs':1,'batch_size':1024}),
        'Seq2Seq':Seq2Seq({'n_epochs':1,'batch_size':1024}),
        
        #'AFHMM_SAC':AFHMM_SAC({}),

        'GRU_DWT':GRU_DWT( {
            "verbose" : 2,
            "training_results_path" : "/home/user/thesis_results/history/",
            "appliances" : {
                "charger" : {
                    "dwt_timewindow" : 12,
                    "dwt_overlap" : 10,
                    "examples_overlap" : 0,
                    "examples_timewindow" : 300,
                    "epochs" : 1,
                    "batch_size" : 1024,
                    "wavelet": 'bior2.2',
                }
            },
            "predicted_column": ("power", "apparent"), 
        }),
        'GACD':GACD({
            "verbose" : 2,
            "training_results_path" : "/home/user/thesis_results/history/",
            "appliances" : {
                "charger" : {
                    "dwt_timewindow" : 12,
                    "dwt_overlap" : 10,
                    "examples_overlap" : 0,
                    "examples_timewindow" : 300,
                    "epochs" : 1,
                    "batch_size" : 1024,
                    "wavelet": 'bior2.2',
                }
            },
            "predicted_column": ("power", "apparent"), 
        })
    },
    'train': {    
        'datasets': {
            'avEiro': {
                'path': '../../../datasets/avEiro_h5/avEiro.h5',
                'buildings': {
                    1: {
                        'start_time': '2020-11-14',
                        #'end_time': '2021-01-15'
                        'end_time': '2020-11-15'
                    }
                }                
            }
        }
    },
    'test': {
        'datasets': {
            'avEiro': {
                'path': '../../../datasets/avEiro_h5/avEiro.h5',
                'buildings': {
                    1: {
                        'start_time': '2021-01-15',
                        #'end_time': '2021-02-05'
                        'end_time': '2021-01-16'
                    }
                }
            }
        },
        'metrics':['mae', 'rmse']
    }
}


api_results_experiment_1 = API(experiment1)

#Get all the results in the experiment and print them.
errors_keys = api_results_experiment_1.errors_keys
errors = api_results_experiment_1.errors
for i in range(len(errors)):
    print (errors_keys[i])
    print (errors[i])
    print ("\n\n")

#Get all methods used in the experiment and save the models
for m in api_results_experiment_1.methods:
    if m in ["GRU_DWT"]:
        api_results_experiment_1.methods[m].save_model("/home/user/thesis_results/models/" + m)