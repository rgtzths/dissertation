import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=RuntimeWarning)

from os.path import join, isfile
from os import listdir

from nilmtk.disaggregate import Disaggregator
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import pywt
import numpy as np
import joblib

import sys
sys.path.insert(1, "../feature_extractors")
from dwt import get_features
from generate_timeseries import generate_main_timeseries, generate_appliance_timeseries


class GradientBoosting(Disaggregator):
    def __init__(self, params):
        self.model = {}
        self.MODEL_NAME = params.get('model_name', 'GradientBoosting')
        self.timeframe = params.get('timeframe', 5)
        self.timestep = params.get('timestep', 2)
        self.overlap = params.get('overlap', 0.5)
        self.interpolate = params.get('interpolate', 'average')
        self.cv = params.get('cv', 0.16)
        self.column = params.get('predicted_column', ("power", "apparent"))
        self.load_model_path = params.get('load_model_folder',None)
        self.waveletname = params.get('waveletname', 'db4')

        if self.load_model_path:
            self.load_model(self.load_model_path)

    def partial_fit(self, train_main, train_appliances, **load_kwargs):

        print("Preparing the Training Data: X")

        train_data = generate_main_timeseries(train_main, False, self.timeframe, self.timestep, self.overlap, self.interpolate)

        train_data = train_data.reshape(train_data.shape[0], int(self.timeframe*60/self.timestep), len(train_main[0].columns.values))
        
        X_train = get_features(train_data, self.waveletname) 

        for app_name, power in train_appliances:
            print("Preparing the Training Data: Y")
            y_train = generate_appliance_timeseries(power, False, self.timeframe, self.timestep, self.overlap, self.column, self.interpolate)
           
            print("Training ", app_name, " in ", self.MODEL_NAME, " model\n", end="\r")

            if app_name in self.model:
                model = self.model[app_name]
            else:
                params = {'n_estimators': 500,
                        'max_depth': 4,
                        'min_samples_split': 5,
                        'learning_rate': 0.01,
                        'loss': 'ls'
                        }
                model = GradientBoostingRegressor(**params)
            
            model.fit(X_train, y_train)

            self.model[app_name] = model
            
    def disaggregate_chunk(self, test_mains):
        test_predictions_list = []

        print("Preparing the Test Data")
        test_data = generate_main_timeseries(test_mains, True, self.timeframe, self.timestep, self.overlap, self.interpolate)
        
        test_data = test_data.reshape(test_data.shape[0], int(self.timeframe*60/self.timestep), len(test_mains[0].columns.values))
        
        X_test = get_features(test_data, self.waveletname) 

        appliance_powers_dict = {}

        for i, app_name in enumerate(self.model):
        
            print("Estimating power demand for '{}' in '{}'\n".format(app_name, self.MODEL_NAME))
            pred = self.model[app_name].predict(X_test)
            
            column = pd.Series(
                    pred, index=test_mains[0].index, name=i)
            appliance_powers_dict[app_name] = column
        
        test_predictions_list.append(pd.DataFrame(appliance_powers_dict, dtype='float32'))

        return test_predictions_list

    def save_model(self, folder_name):
        for app in self.model:
            joblib.dump(self.model[app], join(folder_name, app+".sav"))

    def load_model(self, folder_name):
        app_models = [f for f in listdir(folder_name) if isfile(join(folder_name, f))]
        for app in app_models:
            self.model[app.split(".")[0]] = joblib.load(join(folder_name, app))
