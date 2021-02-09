

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
  'power': {'mains': ['apparent'],'appliance': ['apparent']},
  'sample_rate': 2,
  'appliances': ['heat pump', 'charger'],
  'methods': {"LSTM":LSTM_RNN(10, 2, ("power", "apparent")), "GRU":GRU_RNN(10, 2, ("power", "apparent"))},
  'train': {    
    'datasets': {
        'avEiro': {
            'path': '../../datasets/avEiro_h5/avEiro.h5',
            'buildings': {
                1: {
                    'start_time': '2020-10-01',
                    'end_time': '2021-01-01'
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
                    'start_time': '2021-01-01',
                    'end_time': '2021-02-06'
                    }
                }
            }
        },
        'metrics':['mae', 'rmse']
    }
}


api_results_experiment_1 = API(experiment1)

print(api_results_experiment_1)
