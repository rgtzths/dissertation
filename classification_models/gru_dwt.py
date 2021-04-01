import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=RuntimeWarning)

from os.path import join, isfile
from os import listdir
import math
import json

from keras.models import Sequential, load_model
from keras.layers import Dense, GRU
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import Sequence

from scipy.stats import randint
from sklearn.metrics import matthews_corrcoef, confusion_matrix, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import numpy as np

import sys
sys.path.insert(1, "../feature_extractors")
from generate_timeseries import generate_appliance_timeseries
from matthews_correlation import matthews_correlation
from wt import get_discrete_features


class GRU_DWT():
    def __init__(self, params):
        #Variable that will store the models trained for each appliance.
        self.model = {}
        #Name used to identify the Model Name.
        self.MODEL_NAME = params.get('model_name', 'GRU DWT')
        #Percentage of values used as cross validation data from the training data.
        self.cv = params.get('cv', 0.16)
        #Column used to obtain the classification values (y). 
        self.column = params.get('predicted_column', ("power", "apparent"))
        #If this variable is not None than the class loads the appliance models present in the folder.
        self.load_model_path = params.get('load_model_folder',None)
        #Dictates the ammount of information to be presented during training and regression.
        self.verbose = params.get('verbose', 0)
        
        self.appliances = params.get('appliances', {})
       
        #Decides if the model runs randomsearch
        self.randomsearch = params.get('randomsearch', False)
        #In case of randomsearch = True this variable contains the information
        #to run the randomsearch (hyperparameter range definition).
        self.randomsearch_params = params.get('randomsearch_params', None)
        #Wavelet name used for DWT

        #In case of existing a model path, load every model in that path.
        if self.load_model_path:
            self.load_model(self.load_model_path)

    def partial_fit(self, train_mains, train_appliances):
         #Checks the need to do random search.
        if not self.randomsearch:
            
            #For each appliance to be classified
            for app_name, appliance_power in train_appliances:
                if( self.verbose != 0):
                    print("Preparing Dataset for %s" % app_name)

                appliance_model = self.appliances.get(app_name)

                dwt_timewindow = appliance_model.get("dwt_timewindow", 12)
                dwt_overlap = appliance_model.get("dwt_overlap", 6)
                timestep = appliance_model.get("timestep", 2)
                examples_timewindow = appliance_model.get("examples_timewindow", 300)
                examples_overlap = appliance_model.get("examples_overlap", 150)
                waveletname = appliance_model.get("waveletname", 'db4')
                batch_size = appliance_model.get("batch_size", 500)
                epochs = appliance_model.get("epochs", 300)

                X_train = get_discrete_features(train_mains, waveletname, timestep, dwt_timewindow, dwt_overlap, examples_timewindow, examples_overlap)

                y_train = self.generate_y(appliance_power, timestep, dwt_timewindow, dwt_overlap, examples_timewindow, examples_overlap)

                X_train, X_cv, y_train, y_cv  = train_test_split(X_train, y_train, stratify=y_train, test_size=self.cv, random_state=0)
                
                if self.verbose != 0:
                    print("Training ", app_name, " in ", self.MODEL_NAME, " model\n", end="\r")

                #Checks if the model already exists and if it doesn't creates a new one.          
                if app_name in self.model:
                    model = self.model[app_name]
                else:
                    model = self.create_model((X_train.shape[1], X_train.shape[2]))

                history = model.fit(BalancedGenerator(X_train, y_train, batch_size), 
                        steps_per_epoch=math.ceil(X_train.shape[0]/batch_size),
                        epochs=epochs, 
                        batch_size=batch_size,
                        validation_data=(X_cv, y_cv),
                        verbose=self.verbose
                        )       

                history = json.dumps(history.history)

                f = open("./models/gru_dwt/history.json", "w")
                f.write(history)
                f.close()

                #Stores the trained model.
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

            appliance_model = self.appliances.get(app_name)
            dwt_timewindow = appliance_model.get("dwt_timewindow", 12)
            dwt_overlap = appliance_model.get("dwt_overlap", 6)
            timestep = appliance_model.get("timestep", 2)
            examples_timewindow = appliance_model.get("examples_timewindow", 300)
            examples_overlap = appliance_model.get("examples_overlap", 150)
            waveletname = appliance_model.get("waveletname", 'db4')
            
            X_test = get_discrete_features(test_mains, waveletname, timestep, dwt_timewindow, dwt_overlap, examples_timewindow, examples_overlap)

            y_test = self.generate_y(appliance_power, timestep, dwt_timewindow, dwt_overlap, examples_timewindow, examples_overlap)

            if self.verbose != 0:
                print("Estimating power demand for '{}' in '{}'\n".format(app_name, self.MODEL_NAME))

            pred = self.model[app_name].predict(X_test)
            pred = [ 0 if p < 0.5 else 1 for p in pred ]
        
            y_test = y_test.reshape(len(y_test),)
            
            tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

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
            self.model[app].save(join(folder_name, app + "_" + self.MODEL_NAME + ".h5"))

    def load_model(self, folder_name):
                #Get all the models trained in the given folder and load them.

        app_models = [f for f in listdir(folder_name) if isfile(join(folder_name, f))]
        for app in app_models:
            self.model[app.split(".")[0]] = load_model(join(folder_name, app))

    def create_model(self, input_shape):
        #Creates a specific model.
        model = Sequential()
        model.add(Dense(64, input_shape=input_shape, activation='relu'))
        model.add(GRU(128, return_sequences=True, activation='relu'))
        model.add(GRU(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(units=1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[matthews_correlation])

        return model

    def random_search(self, train_mains, train_appliances):

        #Stores the parameters for the best model for each appliance
        test_results = {}

        for timewindow in self.randomsearch_params['timewindow']:
            for timestep in self.randomsearch_params['timestep']:

                gru_timewindow = self.gru_timewindow.get("heatpump", 60)
                
                #Obtains de X training data acording to the timewindow and timestep
                train_data = generate_main_timeseries(train_mains, False, timewindow, timestep)
                train_data.reshape(train_data.shape[0], int(timewindow/timestep), len(train_main[0].columns.values))

            
                X_train = get_discrete_features(train_data, self.waveletname)

                X_train = X_train[: int(X_train.shape[0]/5) * 5 ]

                X_train = X_train.reshape(int(X_train.shape[0]/5), 5, X_train.shape[1])

                X_train = X_train.reshape(int(X_train.shape[0]/(gru_timewindow / timewindow)), int(gru_timewindow / timewindow), X_train.shape[1])

                #Defines the input shape according to the timewindow and timestep.
                self.randomsearch_params['model']['input_shape'] = [(X_train.shape[1], X_train.shape[2])]

                self.randomsearch_params['model']['n_nodes'] = randint(int(X_train.shape[1] *0.5), int(X_train.shape[1]*2))
                
                for app_name, appliance_power in train_appliances:
                    #Generate the appliance timeseries acording to the timewindow and timestep
                    y_train = generate_appliance_timeseries(appliance_power, True, timewindow, timestep, self.column)

                    y_train = y_train[np.arange(int(gru_timewindow/timewindow) -1, len(y_train), int(gru_timewindow/timewindow))]

                    model = KerasClassifier(build_fn = self.create_model, verbose=0)
                    
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

                    #Store the best result, if it is actualy the best and the X and y used for the final training.
                    if app_name not in test_results :
                        test_results[app_name] = (fitted_model.best_score_, fitted_model.best_params_, timewindow, timestep)

                    elif test_results[app_name][0] < fitted_model.best_score_:
                        test_results[app_name] = (fitted_model.best_score_, fitted_model.best_params_, timewindow, timestep)

        #Train the final model with the best hyperparameters for each appliance
        for app_name in test_results:

            X_train = X_train = generate_main_timeseries(train_mains, False, test_results[app_name][2], test_results[app_name][3])
                        
            X_cv = X_train[int(len(X_train)*(1-self.cv)):]
            X_train = X_train[0:int(len(X_train)*(1-self.cv))]

            y_train = generate_appliance_timeseries(appliance_power, True, test_results[app_name][2], test_results[app_name][3], self.column)

            y_cv = y_train[int(len(y_train)*(1-self.cv)):]
            y_train = y_train[0:int(len(y_train)*(1-self.cv))]

            print("Training ", app_name, " in ", self.MODEL_NAME, " model\n", end="\r")
            model = self.create_model( test_results[app_name][1]['n_nodes'], test_results[app_name][1]['input_shape'])

            model.fit(X_train, y_train, epochs=test_results[app_name][1]['epochs'], batch_size=test_results[app_name][1]['batch_size'], validation_data=(X_cv, y_cv), verbose=self.verbose)

            self.model[app_name] = model

            self.timewindow[app_name] = test_results[app_name][2]

            self.timestep[app_name] = test_results[app_name][3]
            
        return test_results

    def generate_y(self, dfs, timestep, dwt_timewindow, dwt_overlap, examples_timewindow, examples_overlap):
        #train_class = generate_appliance_timeseries(appliance_power, True, dwt_timewindow, timestep, self.column, dwt_overlap)   
        
        y = []
        examples_step = int((examples_timewindow - examples_overlap) / (dwt_timewindow- dwt_overlap))

        step = int((dwt_timewindow-dwt_overlap)/timestep)

        #Starting the conversion (goes through all the dataframes)
        for df in dfs:
            data = []
            
            current_index = step

            while current_index < len(df):
                    
                data.append(1) if df.loc[df.index[current_index], self.column] > 20 else data.append(0)

                current_index += step
            
            current_index = examples_step

            while current_index < len(data):
                y.append(data[current_index])

                current_index += examples_step
            
        return np.array(y)


class BalancedGenerator(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):

        i_0 = np.random.choice( np.where(self.y == 0)[0], size=int(self.batch_size/2), replace = True)

        i_1 = np.random.choice( np.where(self.y == 1)[0], size=int(self.batch_size/2), replace = True)

        i = np.append(i_0, i_1)

        
        return (self.x[i,:,:], self.y[i])