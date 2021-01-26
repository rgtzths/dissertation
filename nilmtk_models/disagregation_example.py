

from nilmtk.disaggregate import CO, FHMMExact, Mean, Hart85

from nilmtk.api import API
import warnings
warnings.filterwarnings("ignore")

from svm import Svm

experiment1 = {
  'power': {'mains': ['apparent'],'appliance': ['apparent']},
  'sample_rate': 1,
  'appliances': ['electric shower heater'],
  'methods': {"CO":CO({}), "Mean":Mean({}),"FHMM_EXACT":FHMMExact({'num_of_states':3}), "Hart85":Hart85({}), "SVM":Svm({})},
  'train': {    
    'datasets': {
        'avEiro': {
            'path': '../converters/avEiro_h5/avEiro.h5',
            'buildings': {
                1: {
                    'start_time': '2020-10-01',
                    'end_time': '2020-10-31'
                    }
                }                
            }
        }
    },
  'test': {
    'datasets': {
        'avEiro': {
            'path': '../converters/avEiro_h5/avEiro.h5',
            'buildings': {
                1: {
                    'start_time': '2020-10-31',
                    'end_time': '2020-11-10'
                    }
                }
            }
        },
        'metrics':['mae', 'rmse']
    }
}


api_results_experiment_1 = API(experiment1)

print(api_results_experiment_1)
