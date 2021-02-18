

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

from svm import Svm
from lstm import LSTM_RNN
from gru import GRU_RNN

experiment1 = {
  'power': {'mains': ['active'],'appliance': ['active']},
  'sample_rate': 6,
  'appliances': ['kettle', 'fridge', 'washing machine', 'microwave', 'dish washer'],
  #'methods': {"CO":CO({}), "Mean":Mean({}),"FHMM_EXACT":FHMMExact({'num_of_states':2}), "Hart85":Hart85({}), "SVM":Svm({}), "LSTM":LSTM_RNN(10, 2, ("power", "apparent")), "GRU":GRU_RNN(10, 2, ("power", "apparent"))},
  #'methods': {"LSTM":LSTM_RNN({"timeframe":10, "timestep": 2,"predicted_column": ("power", "apparent")}), "GRU":GRU_RNN({"timeframe":10, "timestep": 2,"predicted_column": ("power", "apparent")})},
  'methods': {"LSTM": LSTM_RNN({"timeframe":10, "timestep": 6,"predicted_column": ("power", "active"), "overlap":0.5})},
  'train': {    
    'datasets': {
        'UKDale': {
            'path': '../../datasets/ukdale/ukdale.h5',
            'buildings': {
                1: {
                    'start_time': "2015-07-01",
                    'end_time': "2015-07-02"
                },
                2: {
                    'start_time': "2013-05-22",
                    'end_time': "2013-05-23"
                },             
            }
        }
    }
  },
  'test': {
    'datasets': {
        'UKDale': {
            'path': '../../datasets/ukdale/ukdale.h5',
            'buildings': {
                5: {
                    'start_time': "2014-06-29",
                    'end_time': "2014-06-30"
                    }
                }
            }
        },
        'metrics':['mae', 'rmse']
    }
}

api_results_experiment_1 = API(experiment1)

print(api_results_experiment_1)
