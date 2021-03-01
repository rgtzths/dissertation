

from nilmtk.disaggregate import CO, FHMMExact, Mean, Hart85
from nilmtk.api import API

import warnings
warnings.filterwarnings("ignore")
import sys
import logging

sys.stderr = open('./outputs/err.log', 'w')

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s',
                    filename='./outputs/info.log',
                    filemode='w')

from svm import Svm
from lstm import LSTM_RNN
from gru import GRU_RNN



# Classifies Dish Washer, Fridge, Microwave
experiment1 = {
  'power': {'mains': ['apparent'],'appliance': ['active']},
  'sample_rate': 6,
  'appliances': ['fridge', 'microwave', 'dish washer'],
  'methods': {"lstm": LSTM_RNN({"timeframe":10, "timestep": 6,"predicted_column": ("power", "active"), "overlap":0.5, "epochs": 350, "verbose": 0}), 
              "gru": GRU_RNN({"timeframe":10, "timestep": 6,"predicted_column": ("power", "active"), "overlap":0.5, "epochs": 350, "verbose": 0}),
              "co":CO({}), 
              "mean":Mean({}),
              "fhmm_exact":FHMMExact({'num_of_states':2}), 
              "hart85":Hart85({}), 
              "svm":Svm({})
            },
  'train': {    
    'datasets': {
        'UKDale': {
            'path': '../../datasets/ukdale/ukdale.h5',
            'buildings': {
                1: {
                    'start_time': "2013-05-20",
                    'end_time':  None
                },
                2: {
                    'start_time': "2013-05-22",
                    'end_time': None
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
                    'end_time': None
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
  'methods': {"lstm": LSTM_RNN({"timeframe":10, "timestep": 6,"predicted_column": ("power", "active"), "overlap":0.5, "epochs": 350, "verbose": 0}), 
              "gru": GRU_RNN({"timeframe":10, "timestep": 6,"predicted_column": ("power", "active"), "overlap":0.5, "epochs": 350, "verbose": 0}),
              "co":CO({}), 
              "mean":Mean({}),
              "fhmm_exact":FHMMExact({'num_of_states':2}), 
              "hart85":Hart85({}), 
              "svm":Svm({})
            },
  'train': {    
    'datasets': {
        'UKDale': {
            'path': '../../datasets/ukdale/ukdale.h5',
            'buildings': {
                1: {
                    'start_time': "2013-05-20",
                    'end_time':  None
                }             
            }
        }
    }
  },
  'test': {
    'datasets': {
        'UKDale': {
            'path': '../../datasets/ukdale/ukdale.h5',
            'buildings': {
                2: {
                    'start_time': "2013-05-22",
                    'end_time': None
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
  'methods': {"lstm": LSTM_RNN({"timeframe":10, "timestep": 6,"predicted_column": ("power", "active"), "overlap":0.5, "epochs": 350, "verbose": 0}), 
              "gru": GRU_RNN({"timeframe":10, "timestep": 6,"predicted_column": ("power", "active"), "overlap":0.5, "epochs": 350, "verbose": 0}),
              "co":CO({}), 
              "mean":Mean({}),
              "fhmm_exact":FHMMExact({'num_of_states':2}), 
              "hart85":Hart85({}), 
              "svm":Svm({})
            },
  'train': {    
    'datasets': {
        'UKDale': {
            'path': '../../datasets/ukdale/ukdale.h5',
            'buildings': {
                2: {
                    'start_time': "2013-05-22",
                    'end_time':  None
                }
                3: {
                    'start_time': "2013-05-22",
                    'end_time':  None
                }
            
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
                    'end_time': None
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
  'methods': {"lstm": LSTM_RNN({"timeframe":10, "timestep": 6,"predicted_column": ("power", "active"), "overlap":0.5, "epochs": 350, "verbose": 0}), 
              "gru": GRU_RNN({"timeframe":10, "timestep": 6,"predicted_column": ("power", "active"), "overlap":0.5, "epochs": 350, "verbose": 0}),
              "co":CO({}), 
              "mean":Mean({}),
              "fhmm_exact":FHMMExact({'num_of_states':2}), 
              "hart85":Hart85({}), 
              "svm":Svm({})
            },
  'train': {    
    'datasets': {
        'UKDale': {
            'path': '../../datasets/ukdale/ukdale.h5',
            'buildings': {
                2: {
                    'start_time': "2013-05-22",
                    'end_time':  None
                }
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
                    'end_time': None
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
  'methods': {"lstm": LSTM_RNN({"timeframe":10, "timestep": 6,"predicted_column": ("power", "active"), "overlap":0.5, "epochs": 350, "verbose": 0}), 
              "gru": GRU_RNN({"timeframe":10, "timestep": 6,"predicted_column": ("power", "active"), "overlap":0.5, "epochs": 350, "verbose": 0}),
              "co":CO({}), 
              "mean":Mean({}),
              "fhmm_exact":FHMMExact({'num_of_states':2}), 
              "hart85":Hart85({}), 
              "svm":Svm({})
            },
  'train': {    
    'datasets': {
        'UKDale': {
            'path': '../../datasets/ukdale/ukdale.h5',
            'buildings': {
                5: {
                    'start_time': "2014-06-29",
                    'end_time':  "2014-11-01"
                }
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
                    'start_time': "2014-11-01",
                    'end_time': None
                    }
                }   
            }
        },
        'metrics':['mae', 'rmse']
    }
}

# Boiler

experiment6 = {
  'power': {'mains': ['apparent'],'appliance': ['active']},
  'sample_rate': 6,
  'appliances': ['boiler'],
  'methods': {"lstm": LSTM_RNN({"timeframe":10, "timestep": 6,"predicted_column": ("power", "active"), "overlap":0.5, "epochs": 350, "verbose": 0}), 
              "gru": GRU_RNN({"timeframe":10, "timestep": 6,"predicted_column": ("power", "active"), "overlap":0.5, "epochs": 350, "verbose": 0}),
              "co":CO({}), 
              "mean":Mean({}),
              "fhmm_exact":FHMMExact({'num_of_states':2}), 
              "hart85":Hart85({}), 
              "svm":Svm({})
            },
  'train': {    
    'datasets': {
        'UKDale': {
            'path': '../../datasets/ukdale/ukdale.h5',
            'buildings': {
                1: {
                    'start_time': "2012-11-09",
                    'end_time':  "2017-01-01"
                }
            }
        }
    }
  },
  'test': {
    'datasets': {
        'UKDale': {
            'path': '../../datasets/ukdale/ukdale.h5',
            'buildings': {
                1: {
                    'start_time': "2017-01-01",
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

api_results_experiment_2 = API(experiment2)

#Get all methods used in the experiment and save the models
for m in api_results_experiment_2.methods:
        if m not in ["co", "fhmm_exact", "hart85"]:
        api_results_experiment_2.methods[m].save_model(models_folder + m)

#Get all the results in the experiment and print them.
errors_keys = api_results_experiment_2.errors_keys
errors = api_results_experiment_2.errors
for i in range(len(errors)):
    print (errors_keys[i])
    print (errors[i])
    print ("\n\n")

api_results_experiment_3 = API(experiment3)

#Get all methods used in the experiment and save the models
for m in api_results_experiment_3.methods:
        if m not in ["co", "fhmm_exact", "hart85"]:
        api_results_experiment_3.methods[m].save_model(models_folder + m)

#Get all the results in the experiment and print them.
errors_keys = api_results_experiment_3.errors_keys
errors = api_results_experiment_3.errors
for i in range(len(errors)):
    print (errors_keys[i])
    print (errors[i])
    print ("\n\n")

api_results_experiment_4 = API(experiment4)

#Get all methods used in the experiment and save the models
for m in api_results_experiment_4.methods:
    if m not in ["co", "fhmm_exact", "hart85"]:
        api_results_experiment_4.methods[m].save_model(models_folder + m)

#Get all the results in the experiment and print them.
errors_keys = api_results_experiment_4.errors_keys
errors = api_results_experiment_4.errors
for i in range(len(errors)):
    print (errors_keys[i])
    print (errors[i])
    print ("\n\n")


api_results_experiment_5 = API(experiment5)

#Get all methods used in the experiment and save the models
for m in api_results_experiment_5.methods:
    if m not in ["co", "fhmm_exact", "hart85"]:
        api_results_experiment_5.methods[m].save_model(models_folder + m)

#Get all the results in the experiment and print them.
errors_keys = api_results_experiment_5.errors_keys
errors = api_results_experiment_5.errors
for i in range(len(errors)):
    print (errors_keys[i])
    print (errors[i])
    print ("\n\n")

api_results_experiment_6 = API(experiment6)

#Get all methods used in the experiment and save the models
for m in api_results_experiment_6.methods:
    if m not in ["co", "fhmm_exact", "hart85"]:
        api_results_experiment_6.methods[m].save_model(models_folder + m)

#Get all the results in the experiment and print them.
errors_keys = api_results_experiment_6.errors_keys
errors = api_results_experiment_6.errors
for i in range(len(errors)):
    print (errors_keys[i])
    print (errors[i])
    print ("\n\n")


