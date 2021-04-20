
from nilmtk.api import API

import sys
sys.path.insert(1, "../nilmtk-contrib")

from dae import DAE
from seq2point import Seq2Point
from seq2seq import Seq2Seq
from afhmm_sac import AFHMM_SAC
from gru_dwt import GRU_DWT

import warnings
warnings.filterwarnings("ignore")

#Experiment Definition
experiment1 = {
    'power': {'mains': ['apparent'], 'appliance': ['apparent']},
    'sample_rate': 2,
    'appliances': ['heat pump'],
    'methods': {
        'DAE':DAE({'n_epochs':50,'batch_size':1024}),
        'Seq2Point':Seq2Point({'n_epochs':50,'batch_size':1024}),
        'Seq2Seq':Seq2Seq({'n_epochs':50,'batch_size':1024}),
        'AFHMM_SAC':AFHMM_SAC({}),
        'GRU_DWT':GRU_DWT({
            "verbose" : 2,
            "appliances" : {
                "heat pump" : {
                    "epochs" : 2000,
                }
            },
            "predicted_column": ("power", "apparent"), 
        })
    },
    'train': {    
        'datasets': {
            'avEiro': {
                'path': '../../datasets/avEiro_h5/avEiro.h5',
                'buildings': {
                    1: {
                        'start_time': '2020-10-01',
                        'end_time': '2020-12-01'
                        #'end_time': '2020-10-02T12:00'
                    }
                }                
            }
        }
    },
    'test': {
        'datasets': {
            'avEiro': {
                'path': '../../datasets/avEiro_h5/avEiro.h5',
                'buildings': {
                    1: {
                        'start_time': '2021-01-15',
                        'end_time': '2021-02-05'
                        #'end_time': '2021-01-15T12:00'
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