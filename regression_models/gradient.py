import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=RuntimeWarning)

from os.path import join, isfile
from os import listdir

from nilmtk.disaggregate import Disaggregator

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV

import pandas as pd
import pywt
import numpy as np
import joblib

import sys
sys.path.insert(1, "../feature_extractors")
from wt import get_discrete_features
from generate_timeseries import generate_main_timeseries, generate_appliance_timeseries


class GradientBoosting(Disaggregator):
    def __init__(self, params):
        #Variable that will store the models trained for each appliance.
        self.model = {}
        #Name used to identify the Model Name.
        self.MODEL_NAME = params.get('model_name', 'GradientBoosting')
        #Column used to obtain the regression values (y). 
        self.column = params.get('predicted_column', ("power", "apparent"))
        #If this variable is not None than the class loads the appliance models present in the folder.
        self.load_model_path = params.get('load_model_folder',None)
        #Dictates the ammount of information to be presented during training and regression.
        self.verbose = params.get('verbose', 0)
        #Amount of overlap between two consequent feature vectors (increase overlap to increase dataset size and decrease overlap to reduce)
        self.overlap = params.get('overlap', 0.5)
        #When there are measing values you can use 
        #'average' to interpolate the value as the value between the previous recorded value and the next or 
        #'previous to interpolate the value as the previous value
        self.interpolate = params.get('interpolate', 'average')
        #Defines the window of time of each feature vector (in mins)
        self.timewindow = params.get('timewindow', {})
        #Defines the step bertween each reading (in seconds)
        self.timestep = params.get('timestep', {})
        self.params = params.get('params', {})
        #Decides if the model runs grid search
        self.gridsearch = params.get('gridsearch', False)
        #In case of gridsearch = True this variable contains the information
        #to run the gridsearch (hyperparameter range definition).
        self.gridsearch_params = params.get('gridsearch_params', None)
        self.default_params = {'n_estimators': 500,
                                'max_depth': 4,
                                'min_samples_split': 5,
                                'learning_rate': 0.01,
                                'loss': 'ls'
                            }
        self.waveletname = params.get('waveletname', 'db4')

        if self.load_model_path:
            self.load_model(self.load_model_path)

    def partial_fit(self, train_main, train_appliances, **load_kwargs):

        if not self.gridsearch:
            data = {}

            for app, d in train_appliances:
                n = data.get((self.timewindow.get(app, 5), self.timestep.get(app, 6)), 0)
                data[(self.timewindow.get(app, 5), self.timestep.get(app, 6))] = n+1
            
            items_removed = []
            for d in data.keys():
                if data[d] < 2:
                    items_removed.append(d)
            for item in items_removed:
                del data[item]

            for app_name, power in train_appliances:
                print("Preparing Dataset for %s" % app_name)
                timewindow = self.timewindow.get(app_name, 5)
                timestep = self.timestep.get(app_name, 6)

                if (timewindow, timestep) not in data:
                    train_data = generate_main_timeseries(train_main, False, timewindow, timestep, self.overlap, self.interpolate)

                    train_data = train_data.reshape(train_data.shape[0], int(timewindow*60/timestep), len(train_main[0].columns.values))
                    
                    X_train = get_discrete_features(train_data, self.waveletname)

                elif isinstance(data[(timewindow, timestep)], int):
                    train_data = generate_main_timeseries(train_main, False, timewindow, timestep, self.overlap, self.interpolate)

                    train_data = train_data.reshape(train_data.shape[0], int(timewindow*60/timestep), len(train_main[0].columns.values))
                    
                    X_train = get_discrete_features(train_data, self.waveletname)

                    data[(timewindow, timestep)] = X_train
                else:
                    X_train = data[(timewindow, timestep)]

                y_train = generate_appliance_timeseries(power, False, timewindow, timestep, self.overlap, self.column, self.interpolate)
            
                print("Training ", app_name, " in ", self.MODEL_NAME, " model\n", end="\r")

                if app_name in self.model:
                    model = self.model[app_name]
                else:
                    model = GradientBoostingRegressor(**self.params.get(app_name, self.default_params))
                
                model.fit(X_train, y_train)

                self.model[app_name] = model
        else:
            print("Executing RandomSearch")
            results = self.grid_search(train_main, train_appliances)

            for app in results:
                print("\nResults for appliance: ", app)
                print("\t Score Obtained: ", str(results[app][0]))
                print("\t Best Parameters: ", str(results[app][1]))
                print("\t Time Window: ", str(results[app][2]))
                print("\t Time Step: ", str(results[app][3]))
                print("\n")
            
    def disaggregate_chunk(self, test_mains):
        test_predictions_list = []

        data = {}

        for app in self.model:
            n = data.get((self.timewindow.get(app, 5), self.timestep.get(app, 6)), 0)
            data[(self.timewindow.get(app, 5), self.timestep.get(app, 6))] = n+1
        
        items_removed = []
        for d in data.keys():
            if data[d] < 2:
                items_removed.append(d)
        for item in items_removed:
            del data[item]

        appliance_powers_dict = {}

        for i, app_name in enumerate(self.model):
            print("Preparing the Test Data for %s" % app_name)
            timewindow = self.timewindow.get(app_name, 5)
            timestep = self.timestep.get(app_name, 6)
            
            if (timewindow, timestep) not in data:
                test_data = generate_main_timeseries(test_mains, True, timewindow, timestep, self.overlap, self.interpolate)
        
                test_data = test_data.reshape(test_data.shape[0], int(timewindow*60/timestep), len(test_mains[0].columns.values))
                
                X_test = get_discrete_features(test_data, self.waveletname)

            elif isinstance(data[(timewindow, timestep)], int):
                test_data = generate_main_timeseries(test_mains, True, timewindow, timestep, self.overlap, self.interpolate)
        
                test_data = test_data.reshape(test_data.shape[0], int(timewindow*60/timestep), len(test_mains[0].columns.values))
                
                X_test = get_discrete_features(test_data, self.waveletname)

                data[(timewindow, timestep)] = X_test
            else:
                X_test = data[(timewindow, timestep)]

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

    def grid_search(self, train_main, train_appliances):

        #Stores the parameters for the best model for each appliance
        test_results = {}

        #Stores the processed dataset for the best models of each appliance
        data = {}

        for timewindow in self.gridsearch_params['timewindow']:
            for timestep in self.gridsearch_params['timestep']:
                
                #Obtains de X training data acording to the timewindow and timestep
                train_data = generate_main_timeseries(train_main, False, timewindow, timestep, self.overlap, self.interpolate)

                train_data = train_data.reshape(train_data.shape[0], int(timewindow*60/timestep), len(train_main[0].columns.values))
                
                X_train = get_discrete_features(train_data, self.waveletname)
                
                for app_name, power in train_appliances:
                    #Generate the appliance timeseries acording to the timewindow and timestep
                    y_train = generate_appliance_timeseries(power, False, timewindow, timestep, self.overlap, self.column, self.interpolate)

                    model = GradientBoostingRegressor()
                    
                    gridsearch = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=self.gridsearch_params['model'], 
                        cv=10, 
                        n_iter=self.gridsearch_params['n_iter'],
                        n_jobs=self.gridsearch_params['n_jobs'], 
                        scoring='neg_root_mean_squared_error',
                        verbose=2,
                        refit = False
                    )

                    fitted_model = gridsearch.fit(X_train, y_train)

                    #Store the best result, if it is actualu the best and the X and y used for the final training.
                    if app_name not in test_results :
                        test_results[app_name] = (fitted_model.best_score_, fitted_model.best_params_, timewindow, timestep)
                        data[(timewindow, timestep)] = data.get((timewindow, timestep), [X_train, {}])
                        data[(timewindow, timestep)][1][app_name] = y_train

                    elif test_results[app_name][0] < fitted_model.best_score_:
                        test_results[app_name] = (fitted_model.best_score_, fitted_model.best_params_, timewindow, timestep)
                        data[(timewindow, timestep)] = data.get((timewindow, timestep), [X_train, {}])
                        data[(timewindow, timestep)][1][app_name] = y_train

            items_removed = []
            for d in data.keys():
                if not data[d][1]:
                    items_removed.append(d)
            
            for item in items_removed:
                del data[d]

        #Train the final model with the best hyperparameters for each appliance
        for app_name in test_results:

            X_train = data[(test_results[app_name][2], test_results[app_name][3])][0]

            y_train = data[(test_results[app_name][2], test_results[app_name][3])][1][app_name]

            model = GradientBoostingRegressor(**test_results[app_name][1])

            model.fit(X_train, y_train)

            self.model[app_name] = model

            self.timewindow[app_name] = test_results[app_name][2]

            self.timestep[app_name] = test_results[app_name][3]

        return test_results
