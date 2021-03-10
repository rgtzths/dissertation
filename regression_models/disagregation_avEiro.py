

from nilmtk.disaggregate import CO, FHMMExact, Mean, Hart85
from nilmtk.api import API

import warnings
warnings.filterwarnings("ignore")

from svm import Svm
from lstm import LSTM_RNN
from gru import GRU_RNN
from cnn import CNN
from gradient import GradientBoosting

#Experiment Definition
experiment1 = {
  'power': {'mains': ['apparent'], 'appliance': ['apparent']},
  'sample_rate': 2,
  'appliances': ['heat pump'],
  'methods': {"lstm": LSTM_RNN({"timeframe":10, "timestep": 2,"predicted_column": ("power", "apparent"), "overlap":0.5, "epochs": 350, "verbose": 0}), 
              "gru": GRU_RNN({"timeframe":10, "timestep": 2,"predicted_column": ("power", "apparent"), "overlap":0.5, "epochs": 350, "verbose": 0}),
              #"co":CO({}), 
              #"mean":Mean({}),
              #"fhmm_exact":FHMMExact({'num_of_states':2}), 
              #"hart85":Hart85({}), 
              "svm":Svm({}),
              "cnn":CNN({"timeframe":10, "timestep": 2, "predicted_column": ("power", "apparent"), "overlap":0.5, "epochs": 10, "verbose": 0}),
              "gradient_boosting":GradientBoosting({"timeframe":10, "timestep": 2, "predicted_column": ("power", "apparent"), "overlap":0.5})
            },
  'train': {    
    'datasets': {
        'avEiro': {
            'path': '../../datasets/avEiro_h5/avEiro.h5',
            'buildings': {
                1: {
                    'start_time': '2020-10-31',
                    'end_time': '2020-10-31T01:00'
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
                    'end_time': '2021-01-31T01:00'
                    }
                }
            }
        },
        'metrics':['mae', 'rmse']
    }
}

models_folder = "./models/"

api_results_experiment_1 = API(experiment1)

#Get all methods used in the experiment and save the models
for m in api_results_experiment_1.methods:
    if m not in ["co", "fhmm_exact", "hart85"]:
        api_results_experiment_1.methods[m].save_model(models_folder + m)
print ("\n\n")
#Get all the results in the experiment and print them.
errors_keys = api_results_experiment_1.errors_keys
errors = api_results_experiment_1.errors
for i in range(len(errors)):
    print (errors_keys[i])
    print (errors[i])
    print ("\n\n")