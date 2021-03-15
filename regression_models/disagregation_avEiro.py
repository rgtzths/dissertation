

from nilmtk.disaggregate import CO, FHMMExact, Mean, Hart85
from nilmtk.api import API

import warnings
warnings.filterwarnings("ignore")

from svm import Svm
from lstm import LSTM_RNN
from gru import GRU_RNN
from cnn import CNN
from gradient import GradientBoosting

from scipy.stats import randint, loguniform

#Experiment Definition
experiment1 = {
    'power': {'mains': ['apparent'], 'appliance': ['apparent']},
    'sample_rate': 2,
    'appliances': ['heat pump'],
    'methods': {
        "svm":Svm({
            "timewindow": {"heat pump" : 10}, 
            "timestep": {"heat pump" : 2}, 
            "predicted_column": ("power", "apparent"), 
            "params" : {
                "heat pump" : {
                    'kernel': 'rbf',
                    'C': 0.1,
                    'degree': 1,
                    'coef0': 0.1,
                    'epsilon': 0.1,
                    'tol' : 0.0001
                }
            },
            "overlap":0.5,
            "gridsearch" : False,
            "gridsearch_params" : {
                "timewindow" : [5],
                "timestep" : [2],
                "n_iter" : 1,
                "n_jobs" : -1,
                "model": {
                    'kernel': ['rbf', 'poly'],
                    'C': loguniform(0.001, 1),
                    'degree': randint(2,10),
                    'coef0': loguniform(0.001, 1),
                    'epsilon': loguniform(0.001, 1),
                    'tol' : loguniform(0.00001, 0.01)
                }
            }
        }),
        "cnn":CNN( {
            "timewindow": {"heat pump" : 5}, 
            "timestep": {"heat pump" : 2},
            "n_nodes" : {"heat pump" : 300},
            "epochs" : {"heat pump" : 10},
            "batch_size" : {"heat pump" : 500},
            "predicted_column": ("power", "apparent"), 
            "overlap":0.5, 
            "verbose": 0,
            "gridsearch": False,
            "gridsearch_params": {
                "timewindow" : [5],
                "timestep" : [2],
                "n_iter" : 1,
                "n_jobs" : 1,
                "model": {
                    "epochs": randint(10,50),
                    "batch_size" : randint(10,1000),
                }
            }
        }),
        "lstm":LSTM_RNN( {
            "timewindow": {"heat pump" : 10}, 
            "timestep": {"heat pump" : 2},
            "n_nodes" : {"heat pump" : 300},
            "epochs" : {"heat pump" : 10},
            "batch_size" : {"heat pump" : 500},
            "predicted_column": ("power", "apparent"), 
            "overlap":0.5, 
            "verbose": 0,
            "gridsearch": False,
            "gridsearch_params": {
                "timewindow" : [5],
                "timestep" : [2],
                "n_iter" : 1,
                "n_jobs" : -1,
                "model": {
                    "epochs": randint(10,50),
                    "batch_size" : randint(10,1000),
                }
            }
        }),
        "gru":GRU_RNN( {
            "timewindow": {"heat pump" : 10}, 
            "timestep": {"heat pump" : 2},
            "n_nodes" : {"heat pump" : 300},
            "epochs" : {"heat pump" : 10},
            "batch_size" : {"heat pump" : 500},
            "predicted_column": ("power", "apparent"), 
            "overlap":0.5, 
            "verbose": 0,
            "gridsearch": False,
            "gridsearch_params": {
                "timewindow" : [5],
                "timestep" : [2],
                "n_iter" : 1,
                "n_jobs" : -1,
                "model": {
                    "epochs": randint(10,50),
                    "batch_size" : randint(10,1000),
                }
            }
        }),
        "gradient_boosting": GradientBoosting( {
            "timewindow": {"heat pump" : 10}, 
            "timestep": {"heat pump" : 2}, 
            "predicted_column": ("power", "apparent"), 
            "params" : {
                "heat pump" : {
                    'n_estimators': 500,
                    'max_depth': 4,
                    'min_samples_split': 5,
                    'learning_rate': 0.01,
                    'loss': 'ls'
                }
            },
            "overlap":0.5,
            "gridsearch" : False,
            "gridsearch_params" : {
                "timewindow" : [5],
                "timestep" : [2],
                "n_iter" : 1,
                "n_jobs" : -1,
                "model": {
                    'n_estimators': randint(200,1000),
                    'max_depth': randint(2,10),
                    'min_samples_split': randint(2,10),
                    'learning_rate': loguniform(0.001, 1),
                    'loss': ['ls']
                }
            }
        })
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