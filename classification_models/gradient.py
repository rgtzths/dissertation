import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=RuntimeWarning)

from os.path import join, isfile
from os import listdir

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import matthews_corrcoef, confusion_matrix
import pywt
import numpy as np

import joblib
import sys
sys.path.insert(1, "../feature_extractors")

from dwt import get_features
import dataset_loader
import generate_timeseries

class GradientBoosting():
    def __init__(self, params):
        self.MODEL_NAME = params.get('model_name', 'GradientBoosting')
        self.model = {}
        self.timeframe = params.get('timeframe', None)
        self.timestep = params.get('timestep', 2)
        self.overlap = params.get('overlap', 0.5)
        self.interpolate = params.get('interpolate', 'average')
        self.column = params.get('predicted_column', ("power", "apparent"))
        self.cv = params.get('cv', 0.16)
        self.load_model_path = params.get('load_model_folder',None)
        self.epochs = params.get('epochs', 300)
        self.verbose = params.get('verbose', 0)
        self.waveletname = params.get('waveletname', 'db4')
        
        if self.load_model_path:
            self.load_model(self.load_model_path)

    def partial_fit(self, train_main, train_appliance, app):

        train_data = generate_timeseries.generate_main_timeseries(train_main, False, self.timeframe, self.timestep, self.overlap, self.interpolate)
        y_train = generate_timeseries.generate_appliance_timeseries(train_appliance, True, self.timeframe, self.timestep, self.overlap, self.column, self.interpolate)

        train_data = train_data.reshape(train_data.shape[0], int(self.timeframe*60/self.timestep), len(train_main[0].columns.values))
        
        X_train = get_features(train_data, self.waveletname)
        
        if app in self.model:
            model = self.model[app]
        else:
            params = {'n_estimators': 500,
                        'max_depth': 4,
                        'min_samples_split': 5,
                        'learning_rate': 0.01,
                        }
            model = GradientBoostingClassifier(**params)
        y_train[0] = 1
        y_train[1] = 0
        model.fit(X_train, y_train)
        
        self.model[app] = model
            
    def disaggregate_chunk(self, test_main, test_appliance, app):
        test_data = generate_timeseries.generate_main_timeseries(test_main, False, self.timeframe, self.timestep, self.overlap, self.interpolate)
        y_test = generate_timeseries.generate_appliance_timeseries(test_appliance, True, self.timeframe, self.timestep, self.overlap, self.column, self.interpolate)

        test_data = test_data.reshape(test_data.shape[0], int(self.timeframe*60/self.timestep), len(test_main[0].columns.values))

        X_test = get_features(test_data, self.waveletname)


        pred = self.model[app].predict(X_test)
                
        #tn, fn, fp, tp = confusion_matrix(y_test, pred).ravel()
        
        #print("True Positives: ", tp)
        #print("True Negatives: ", tn)  
        #print("False Negatives: ", fn)  
        #print("False Positives: ", fp)        
        #print( "MCC: ", matthews_corrcoef(y_test, pred))

        return matthews_corrcoef(y_test, pred)

    def save_model(self, folder_name):
        for app in self.model:
            joblib.dump(self.model[app], join(folder_name, app+".sav"))

    def load_model(self, folder_name):
        app_models = [f for f in listdir(folder_name) if isfile(join(folder_name, f))]
        for app in app_models:
            self.model[app.split(".")[0]] = joblib.load(join(folder_name, app))
