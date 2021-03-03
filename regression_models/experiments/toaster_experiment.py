import sys
sys.path.insert(1, "..")

from nilmtk.disaggregate import CO, FHMMExact, Mean, Hart85
from nilmtk.api import API

import warnings
warnings.filterwarnings("ignore")
import logging

sys.stderr = open('./toaster_err.log', 'w')

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filename='./toaster_info.log',
                    filemode='w')

from lstm import LSTM_RNN
from gru import GRU_RNN

#Experiment Definition
experiment1 = {
  'power': {'mains': ['apparent'],'appliance': ['active']},
  'sample_rate': 6,
  'appliances': ['toaster'],
  'methods': {
        "LSTM_1_0" : LSTM_RNN( {
            "timeframe" : 1,
            "timestep" : 6,
            "predicted_column": ("power", "active"),
            "overlap" : 0.5,
            "interpolate" : "average",
            "epochs" : 400,
            "input_size" : 0,
            "model_name" : "LSTM_1_0"
        }),
        "LSTM_5_0" : LSTM_RNN( {
            "timeframe" : 5,
            "timestep" : 6,
            "predicted_column": ("power", "active"),
            "overlap" : 0.5,
            "interpolate" : "average",
            "epochs" : 400,
            "input_size" : 0,
            "model_name" : "LSTM_5_0"
        }),
        "LSTM_10_0" : LSTM_RNN( {
            "timeframe" : 10,
            "timestep" : 6,
            "predicted_column": ("power", "active"),
            "overlap" : 0.5,
            "interpolate" : "average",
            "epochs" : 400,
            "input_size" : 0,
            "model_name" : "LSTM_10_0"
        }),
        "LSTM_30_0" : LSTM_RNN( {
            "timeframe" : 30,
            "timestep" : 6,
            "predicted_column": ("power", "active"),
            "overlap" : 0.5,
            "interpolate" : "average",
            "epochs" : 400,
            "input_size" : 0,
            "model_name" : "LSTM_30_0"
        }),
        "LSTM_60_0" : LSTM_RNN( {
            "timeframe" : 60,
            "timestep" : 6,
            "predicted_column": ("power", "active"),
            "overlap" : 0.5,
            "interpolate" : "average",
            "epochs" : 400,
            "input_size" : 0,
            "model_name" : "LSTM_60_0"
        }),
        "LSTM_120_0" : LSTM_RNN( {
            "timeframe" : 10,
            "timestep" : 6,
            "predicted_column": ("power", "active"),
            "overlap" : 0.5,
            "interpolate" : "average",
            "epochs" : 400,
            "input_size" : 0,
            "model_name" : "LSTM_120_0"
        }),
        "GRU_1_0" : GRU_RNN( {
            "timeframe" : 1,
            "timestep" : 6,
            "predicted_column": ("power", "active"),
            "overlap" : 0.5,
            "interpolate" : "average",
            "epochs" : 400,
            "input_size" : 0,
            "model_name" : "GRU_1_0"
        }),
        "GRU_5_0" : GRU_RNN( {
            "timeframe" : 5,
            "timestep" : 6,
            "predicted_column": ("power", "active"),
            "overlap" : 0.5,
            "interpolate" : "average",
            "epochs" : 400,
            "input_size" : 0,
            "model_name" : "GRU_5_0"
        }),
        "GRU_10_0" : GRU_RNN( {
            "timeframe" : 10,
            "timestep" : 6,
            "predicted_column": ("power", "active"),
            "overlap" : 0.5,
            "interpolate" : "average",
            "epochs" : 400,
            "input_size" : 0,
            "model_name" : "GRU_10_0"
        }),
        "GRU_30_0" : GRU_RNN( {
            "timeframe" : 30,
            "timestep" : 6,
            "predicted_column": ("power", "active"),
            "overlap" : 0.5,
            "interpolate" : "average",
            "epochs" : 400,
            "input_size" : 0,
            "model_name" : "GRU_30_0"
        }),
        "GRU_60_0" : GRU_RNN( {
            "timeframe" : 60,
            "timestep" : 6,
            "predicted_column": ("power", "active"),
            "overlap" : 0.5,
            "interpolate" : "average",
            "epochs" : 400,
            "input_size" : 0,
            "model_name" : "GRU_60_0"
        }),
        "GRU_120_0" : GRU_RNN( {
            "timeframe" : 10,
            "timestep" : 6,
            "predicted_column": ("power", "active"),
            "overlap" : 0.5,
            "interpolate" : "average",
            "epochs" : 400,
            "input_size" : 0,
            "model_name" : "GRU_120_0"
        }),
    },
    'train': {    
        'datasets': {
            'UKDale': {
                'path': '../../../datasets/ukdale/ukdale.h5',
                'buildings': {
                    2: {
                        'start_time': None,
                        'end_time':  None
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
                        'start_time': None,
                        'end_time': None
                        }
                    }
                }
            },
            'metrics':['mae', 'rmse']
    }
}

models_folder = "../models/"

api_results_experiment_1 = API(experiment1)

#Get all methods used in the experiment and save the models
for m in api_results_experiment_1.methods:
    if m not in ["co", "fhmm_exact", "hart85"]:
        api_results_experiment_1.methods[m].save_model(models_folder + m)

#Get all the results in the experiment and print them.
errors_keys = api_results_experiment_1.errors_keys
errors = api_results_experiment_1.errors
for i in range(len(errors)):
    print (errors_keys[i])
    print (errors[i])
    print ("\n\n")