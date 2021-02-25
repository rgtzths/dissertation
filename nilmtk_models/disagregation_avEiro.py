

from nilmtk.disaggregate import CO, FHMMExact, Mean, Hart85
from nilmtk.api import API

import warnings
warnings.filterwarnings("ignore")
import sys
import logging

sys.stderr = open('./err.log', 'w')

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filename='./info.log',
                    filemode='w')

from svm import Svm
from lstm import LSTM_RNN
from gru import GRU_RNN

experiment1 = {
  'power': {'mains': ['apparent'], 'appliance': ['apparent']},
  'sample_rate': 2,
  'appliances': ['heat pump', 'charger'],
  'methods': {"lstm": LSTM_RNN({"timeframe":10, "timestep": 2,"predicted_column": ("power", "apparent"), "overlap":0.5, "epochs": 350, "verbose": 0}), 
              "gru": GRU_RNN({"timeframe":10, "timestep": 2,"predicted_column": ("power", "apparent"), "overlap":0.5, "epochs": 350, "verbose": 0}),
              "co":CO({}), 
              "mean":Mean({}),
              "fhmm_exact":FHMMExact({'num_of_states':2}), 
              "hart85":Hart85({}), 
              "svm":Svm({})
            },
  'train': {    
    'datasets': {
        'avEiro': {
            'path': '../../datasets/avEiro_h5/avEiro.h5',
            'buildings': {
                1: {
                    'start_time': '2020-10-01',
                    'end_time': '2021-01-31'
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
                    'start_time': '2021-01-31',
                    'end_time': None
                    }
                }
            }
        },
        'metrics':['mae', 'rmse']
    }
}

models_folder = "./models/"

api_results_experiment_1 = API(experiment1)

for m in api_results_experiment_1.methods:
    if m not in ["co", "fhmm_exact", "hart85"]:
        api_results_experiment_1.methods[m].save_model(models_folder + m)


errors_keys = api_results_experiment_1.errors_keys
errors = api_results_experiment_1.errors
for i in range(len(errors)):
    print (errors_keys[i])
    print (errors[i])
    print ("\n\n")