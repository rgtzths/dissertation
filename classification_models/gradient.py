import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=RuntimeWarning)

from os.path import join, isfile
from os import listdir

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import matthews_corrcoef, confusion_matrix, make_scorer

import numpy as np

import joblib
import sys
sys.path.insert(1, "../feature_extractors")

from wt import get_discrete_features
from generate_timeseries import generate_main_timeseries, generate_appliance_timeseries

class GradientBoosting():
    def __init__(self, params):
        #Variable that will store the models trained for each appliance.
        self.model = {}
        #Name used to identify the Model Name.
        self.MODEL_NAME = params.get('model_name', 'GradientBoosting')
        #Column used to obtain the classification values (y). 
        self.column = params.get('predicted_column', ("power", "apparent"))
        #If this variable is not None than the class loads the appliances model present in the folder.
        self.load_model_path = params.get('load_model_folder',None)
        #Dictates the ammount of information to be presented during training and regression.
        self.verbose = params.get('verbose', 0)
        #Defines the window of time of each feature vector (in mins)
        self.timewindow = params.get('timewindow', {})
        #Defines the step bertween each reading (in seconds)
        self.timestep = params.get('timestep', {})
        #Defines the model params for each appliance
        self.params = params.get('params', {})

        #Decides if the model runs random search
        self.randomsearch = params.get('randomsearch', False)
        #In case of randomsearch = True this variable contains the information
        #to run the randomsearch (hyperparameter range definition).
        self.randomsearch_params = params.get('randomsearch_params', None)
        
        #Default params for the model if no params are defined
        self.default_params = {
            'n_estimators': 500,
            'max_depth': 4,
            'min_samples_split': 5,
            'learning_rate': 0.01,
            'loss': 'deviance'
        }
        #Wavelet name used for DWT
        self.waveletname = params.get('waveletname', 'db4')

        #If the path is not None loads every appliance model in the folder.
        if self.load_model_path:
            self.load_model(self.load_model_path)
    
    def partial_fit(self, train_mains, train_appliances):

        #Checks the need to do random search.
        if not self.randomsearch:

            #For each appliance to be classified
            for app_name, appliance_power in train_appliances:
                if( self.verbose != 0):
                    print("Preparing Dataset for %s" % app_name)
                
                #Get the timewindow and timestep
                timewindow = self.timewindow.get(app_name, 5)
                timestep = self.timestep.get(app_name, 6)

                train_data = generate_main_timeseries(train_mains, False, timewindow, timestep)
                
                X_train = get_discrete_features(train_data, self.waveletname)

                y_train = generate_appliance_timeseries(appliance_power, True, timewindow, timestep, self.column)
            
                if self.verbose != 0:
                    print("Training ", app_name, " in ", self.MODEL_NAME, " model\n", end="\r")

                #Checks if the model already exists and if it doesn't creates a new one.
                if app_name in self.model:
                    model = self.model[app_name]
                else:
                    model = GradientBoostingClassifier(**self.params.get(app_name, self.default_params))
                #Fits the model to the training data.
                model.fit(X_train, y_train)
                #Stores the trained model
                self.model[app_name] = model
        else:
            if self.verbose != 0:
                print("Executing RandomSearch")
            results = self.random_search(train_mains, train_appliances)

            if self.verbose == 2:
                for app in results:
                    print("\nResults for appliance: ", app)
                    print("\t Score Obtained: ", str(results[app][0]))
                    print("\t Best Parameters: ", str(results[app][1]))
                    print("\t Time Window: ", str(results[app][2]))
                    print("\t Time Step: ", str(results[app][3]))
                    print("\n")
            
    def disaggregate_chunk(self, test_mains, test_appliances):

        appliance_powers_dict = {}

        for app_name, appliance_power in test_appliances:
            if self.verbose != 0:
                print("Preparing the Test Data for %s" % app_name)

            timewindow = self.timewindow.get(app_name, 5)
            timestep = self.timestep.get(app_name, 6)
            
            test_data = generate_main_timeseries(test_mains, False, timewindow, timestep)
                    
            X_test = get_discrete_features(test_data, self.waveletname)
        
            y_test = generate_appliance_timeseries(appliance_power, True, timewindow, timestep, self.column)

            if self.verbose != 0:
                print("Estimating power demand for '{}' in '{}'\n".format(app_name, self.MODEL_NAME))

            pred = self.model[app_name].predict(X_test)
                
            tn, fn, fp, tp = confusion_matrix(y_test, pred).ravel()
            
            if self.verbose == 2:
                print("True Positives: ", tp)
                print("True Negatives: ", tn)  
                print("False Negatives: ", fn)  
                print("False Positives: ", fp)        
                print( "MCC: ", matthews_corrcoef(y_test, pred))

            appliance_powers_dict[app_name] = matthews_corrcoef(y_test, pred)
        return appliance_powers_dict

    def save_model(self, folder_name):
        #For each appliance trained store its model
        for app in self.model:
            joblib.dump(self.model[app], join(folder_name, app+".sav"))

    def load_model(self, folder_name):
        #Get all the models trained in the given folder and load them.
        app_models = [f for f in listdir(folder_name) if isfile(join(folder_name, f))]
        for app in app_models:
            self.model[app.split(".")[0]] = joblib.load(join(folder_name, app))

    def random_search(self, train_mains, train_appliances):

        #Stores the parameters for the best model for each appliance
        test_results = {}

        for timewindow in self.randomsearch_params['timewindow']:
            for timestep in self.randomsearch_params['timestep']:
                
                #Obtains de X training data acording to the timewindow and timestep
                train_data = generate_main_timeseries(train_mains, False, timewindow, timestep)

                X_train = get_discrete_features(train_data, self.waveletname)
                
                for app_name, appliance_power in train_appliances:
                    #Generate the appliance timeseries acording to the timewindow and timestep
                    y_train = generate_appliance_timeseries(appliance_power, True, timewindow, timestep, self.column)

                    model = GradientBoostingClassifier()
                    
                    randomsearch = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=self.randomsearch_params['model'], 
                        cv=self.randomsearch_params['cv'], 
                        n_iter=self.randomsearch_params['n_iter'],
                        n_jobs=self.randomsearch_params['n_jobs'], 
                        scoring=make_scorer(matthews_corrcoef),
                        verbose=self.verbose,
                        refit = False
                    )

                    fitted_model = randomsearch.fit(X_train, y_train)

                    #Store the best result, if it is actualu the best and the X and y used for the final training.
                    if app_name not in test_results :
                        test_results[app_name] = (fitted_model.best_score_, fitted_model.best_params_, timewindow, timestep)

                    elif test_results[app_name][0] < fitted_model.best_score_:
                        test_results[app_name] = (fitted_model.best_score_, fitted_model.best_params_, timewindow, timestep)

        #Train the final model with the best hyperparameters for each appliance
        for app_name in test_results:
            
            train_data = generate_main_timeseries(train_mains, False, test_results[app_name][2], test_results[app_name][3])

            X_train = get_discrete_features(train_data, self.waveletname)


            y_train = generate_appliance_timeseries(appliance_power, True, test_results[app_name][2], test_results[app_name][3], self.column)

            print("Training ", app_name, " in ", self.MODEL_NAME, " model\n", end="\r")
            model = GradientBoostingClassifier(**test_results[app_name][1])

            model.fit(X_train, y_train)

            self.model[app_name] = model

            self.timewindow[app_name] = test_results[app_name][2]

            self.timestep[app_name] = test_results[app_name][3]

        return test_results
