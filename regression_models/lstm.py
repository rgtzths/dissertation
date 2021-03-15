import warnings
warnings.filterwarnings('ignore')

from os.path import join, isfile
from os import listdir

from nilmtk.disaggregate import Disaggregator

from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

import pandas as pd

import sys
sys.path.insert(1, "../feature_extractors")
from generate_timeseries import generate_main_timeseries, generate_appliance_timeseries

class LSTM_RNN(Disaggregator):
    def __init__(self, params):
        #Variable that will store the models trained for each appliance.
        self.model = {}
        #Name used to identify the Model Name.
        self.MODEL_NAME = params.get('model_name', 'LSTM')
        #Percentage of values used as cross validation data from the training data.
        self.cv = params.get('cv', 0.16)
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
        #Defines the number of nodes in the LSTM, 
        #its a percentage of the number of values per feature vector
        #Usualy 0.9, 1 or 1.1 of the input size
        self.n_nodes = params.get('n_nodes', {})
        #Number of epochs that the models run
        self.epochs = params.get('epochs', {})
        #Number of examples presented per batch
        self.batch_size = params.get('batch_size', 1000)
        #Decides if the model runs grid search
        self.gridsearch = params.get('gridsearch', False)
        #In case of gridsearch = True this variable contains the information
        #to run the gridsearch (hyperparameter range definition).
        self.gridsearch_params = params.get('gridsearch_params', None)

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
                    X_train = generate_main_timeseries(train_main, False, timewindow, timestep, self.overlap, self.interpolate)

                    X_train = X_train.reshape(X_train.shape[0], int(timewindow*60/timestep), len(train_main[0].columns.values))

                elif isinstance(data[(timewindow, timestep)], int):
                    X_train = generate_main_timeseries(train_main, False, timewindow, timestep, self.overlap, self.interpolate)

                    X_train = X_train.reshape(X_train.shape[0], int(timewindow*60/timestep), len(train_main[0].columns.values))

                    data[(timewindow, timestep)] = X_train
                else:
                    X_train = data[(timewindow, timestep)]
                
                X_cv = X_train[int(len(X_train)*(1-self.cv)):]
                
                X_train = X_train[0:int(len(X_train)*(1-self.cv))]

                y_train = generate_appliance_timeseries(power, False, timewindow, timestep, self.overlap, self.column, self.interpolate)
                
                y_cv = y_train[int(len(y_train)*(1-self.cv)):]
                
                y_train = y_train[0:int(len(y_train)*(1-self.cv))]
            
                print("Training ", app_name, " in ", self.MODEL_NAME, " model\n", end="\r")
                
                if app_name in self.model:
                    model = self.model[app_name]
                else:
                    model = self.create_model(self.n_nodes.get(app_name, X_train.shape[1]),(X_train.shape[1], X_train.shape[2]))

                model.fit(X_train, y_train, epochs=self.epochs.get(app_name, 200), batch_size=self.batch_size.get(app_name, 500), validation_data=(X_cv, y_cv), verbose=self.verbose, shuffle=False)

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
                X_test = generate_main_timeseries(test_mains, True, timewindow, timestep, self.overlap, self.interpolate)
        
                X_test = X_test.reshape(X_test.shape[0], int(timewindow*60/timestep), len(test_mains[0].columns.values))

            elif isinstance(data[(timewindow, timestep)], int):
                X_test = generate_main_timeseries(test_mains, True, timewindow, timestep, self.overlap, self.interpolate)
        
                X_test = X_test.reshape(X_test.shape[0], int(timewindow*60/timestep), len(test_mains[0].columns.values))

                data[(timewindow, timestep)] = X_test
            else:
                X_test = data[(timewindow, timestep)]

            print("Estimating power demand for '{}' in '{}'\n".format(app_name, self.MODEL_NAME))
            pred = self.model[app_name].predict(X_test)
            pred = [p[0] for p in pred]

            column = pd.Series(
                    pred, index=test_mains[0].index, name=i)
            appliance_powers_dict[app_name] = column
        
        test_predictions_list.append(pd.DataFrame(appliance_powers_dict, dtype='float32'))

        return test_predictions_list

    def save_model(self, folder_name):
        for app in self.model:
            self.model[app].save(join(folder_name, app + "_" + self.MODEL_NAME + ".h5"))

    def load_model(self, folder_name):
        app_models = [f for f in listdir(folder_name) if isfile(join(folder_name, f))]
        for app in app_models:
            self.model[app.split(".")[0]] = load_model(join(folder_name, app))


    def create_model(self,n_nodes, input_shape):
        model = Sequential()
        model.add(LSTM(n_nodes, input_shape=input_shape ))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=["RootMeanSquaredError"])

        return model

    def grid_search(self, train_main, train_appliances):

        #Stores the parameters for the best model for each appliance
        test_results = {}

        #Stores the processed dataset for the best models of each appliance
        data = {}

        for timewindow in self.gridsearch_params['timewindow']:
            for timestep in self.gridsearch_params['timestep']:
                
                #Obtains de X training data acording to the timewindow and timestep
                X_train = generate_main_timeseries(train_main, False, timewindow, timestep, self.overlap, self.interpolate)

                X_train = X_train.reshape(X_train.shape[0], int(timewindow*60/timestep), len(train_main[0].columns.values))

                #Defines the input shape according to the timewindow and timestep.
                self.gridsearch_params['model']['input_shape'] = [(X_train.shape[1], X_train.shape[2])]

                self.gridsearch_params['model']['n_nodes'] = randint(int(X_train.shape[1] *0.5), int(X_train.shape[1]*2))
                
                for app_name, power in train_appliances:
                    #Generate the appliance timeseries acording to the timewindow and timestep
                    y_train = generate_appliance_timeseries(power, False, timewindow, timestep, self.overlap, self.column, self.interpolate)

                    model = KerasClassifier(build_fn = self.create_model, verbose=0)
                    
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
                        
            X_cv = X_train[int(len(X_train)*(1-self.cv)):]
            X_train = X_train[0:int(len(X_train)*(1-self.cv))]

            y_train = data[(test_results[app_name][2], test_results[app_name][3])][1][app_name]

            y_cv = y_train[int(len(y_train)*(1-self.cv)):]
            y_train = y_train[0:int(len(y_train)*(1-self.cv))]

            model = self.create_model(test_results[app_name][1]["n_nodes"], test_results[app_name][1]["input_shape"])

            model.fit(X_train, y_train, epochs=test_results[app_name][1]['epochs'], batch_size=test_results[app_name][1]['batch_size'], validation_data=(X_cv, y_cv), verbose=self.verbose, shuffle=True)

            self.model[app_name] = model

            self.timewindow[app_name] = test_results[app_name][2]

            self.timestep[app_name] = test_results[app_name][3]

        return test_results
        