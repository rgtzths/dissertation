import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=RuntimeWarning)

from os.path import join, isfile
from os import listdir
import math
import json

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, GRU
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.utils import Sequence

from scipy.stats import randint
from sklearn.metrics import matthews_corrcoef, confusion_matrix, make_scorer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import sys
import csv

sys.path.insert(1, "../../feature_extractors")
from generate_timeseries import generate_appliance_timeseries
from matthews_correlation import matthews_correlation
from wt import get_discrete_features


class GRU_DWT():
    def __init__(self, params):
        #Variable that will store the models trained for each appliance.
        self.model = {}
        #Name used to identify the Model Name.
        self.MODEL_NAME = params.get('model_name', 'GRU_DWT')
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

        self.default_appliance = {
            "dwt_timewindow": 12,
            "dwt_overlap": 6,
            "timestep": 2,
            "examples_timewindow": 300,
            "examples_overlap": 150,
            "wavelet": 'db4',
            "batch_size": 500,
            "epochs": 300,
            "n_nodes":32
        }

        self.training_results_path = params.get("training_results_path", None)

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

                dwt_timewindow = appliance_model.get("dwt_timewindow", self.default_appliance['dwt_timewindow'])
                dwt_overlap = appliance_model.get("dwt_overlap", self.default_appliance['dwt_overlap'])
                timestep = appliance_model.get("timestep", self.default_appliance['timestep'])
                examples_timewindow = appliance_model.get("examples_timewindow", self.default_appliance['examples_timewindow'])
                examples_overlap = appliance_model.get("examples_overlap", self.default_appliance['examples_overlap'])
                wavelet = appliance_model.get("wavelet", self.default_appliance['wavelet'])
                batch_size = appliance_model.get("batch_size", self.default_appliance['batch_size'])
                epochs = appliance_model.get("epochs", self.default_appliance['epochs'])
                n_nodes = appliance_model.get("n_nodes", self.default_appliance['n_nodes'])

                X_train = get_discrete_features(train_mains, wavelet, timestep, dwt_timewindow, dwt_overlap, examples_timewindow, examples_overlap)

                y_train = self.generate_y(appliance_power, timestep, dwt_timewindow, dwt_overlap, examples_timewindow, examples_overlap)

                X_train, X_cv, y_train, y_cv  = train_test_split(X_train, y_train, stratify=y_train, test_size=self.cv, random_state=0)
                
                if self.verbose != 0:
                    print("Training ", app_name, " in ", self.MODEL_NAME, " model\n", end="\r")

                #Checks if the model already exists and if it doesn't creates a new one.          
                if app_name in self.model:
                    model = self.model[app_name]
                else:
                    model = self.create_model(n_nodes, (X_train.shape[1], X_train.shape[2]))

                history = model.fit(BalancedGenerator(X_train, y_train, batch_size), 
                        steps_per_epoch=math.ceil(X_train.shape[0]/batch_size),
                        epochs=epochs, 
                        batch_size=batch_size,
                        validation_data=(X_cv, y_cv),
                        verbose=self.verbose
                        )       

                history = json.dumps(history.history)

                if self.training_results_path is not None:
                    f = open(self.training_results_path + "history_"+app_name+".json", "w")
                    f.write(history)
                    f.close()

                #Stores the trained model.
                self.model[app_name] = model
        else:
            if self.verbose != 0:
                print("Executing RandomSearch")
            results = self.random_search(train_mains, train_appliances)

    def disaggregate_chunk(self, test_mains, test_appliances):

        appliance_powers_dict = {}
        
        for app_name, appliance_power in test_appliances:
            if self.verbose != 0:
                print("Preparing the Test Data for %s" % app_name)

            appliance_model = self.appliances.get(app_name)
            dwt_timewindow = appliance_model.get("dwt_timewindow", self.default_appliance['dwt_timewindow'])
            dwt_overlap = appliance_model.get("dwt_overlap", self.default_appliance['dwt_overlap'])
            timestep = appliance_model.get("timestep", self.default_appliance['timestep'])
            examples_timewindow = appliance_model.get("examples_timewindow", self.default_appliance['examples_timewindow'])
            examples_overlap = appliance_model.get("examples_overlap", self.default_appliance['examples_overlap'])
            wavelet = appliance_model.get("wavelet", self.default_appliance['wavelet'])
            
            X_test = get_discrete_features(test_mains, wavelet, timestep, dwt_timewindow, dwt_overlap, examples_timewindow, examples_overlap)

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

    def create_model(self, n_nodes, input_shape):
        #Creates a specific model.
        model = Sequential()
        model.add(Dense(n_nodes, input_shape=input_shape, activation='relu'))
        model.add(GRU(n_nodes*2, return_sequences=True, activation='relu'))
        model.add(GRU(n_nodes*2, activation='relu'))
        model.add(Dense(n_nodes, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[matthews_correlation])

        return model

    def random_search(self, train_mains, train_appliances):

        for app_name, appliance_power in train_appliances:
            appliance_model = self.appliances.get(app_name)
                        
            timestep = appliance_model.get("timestep", self.default_appliance['timestep'])

            dwt_timewindow = appliance_model.get("dwt_timewindow", self.default_appliance['dwt_timewindow'])
            dwt_overlap = appliance_model.get("dwt_overlap", self.default_appliance['dwt_overlap'])
            
            examples_timewindow = appliance_model.get("examples_timewindow", self.default_appliance['examples_timewindow'])
            examples_overlap = appliance_model.get("examples_overlap", self.default_appliance['examples_overlap'])
            
            wavelet = appliance_model.get("wavelet", self.default_appliance['wavelet'])

            print("Loading Data")
            X_train = get_discrete_features(train_mains, wavelet, timestep, dwt_timewindow, dwt_overlap, examples_timewindow, examples_overlap)
            
            #Defines the input shape according to the timewindow and timestep.
            self.randomsearch_params['model']['input_shape'] = [(X_train.shape[1], X_train.shape[2])]

            #Generate the appliance timeseries acording to the timewindow and timestep
            y_train = self.generate_y(appliance_power, timestep, dwt_timewindow, dwt_overlap, examples_timewindow, examples_overlap)

            X_train_rc, y_train_rc = shuffle(X_train, y_train, random_state=0)

            model = KerasClassifier(build_fn=self.create_model, verbose=0)
            print("Doing the iterations...")
            randomsearch = RandomizedSearchCV(
                estimator=model,
                param_distributions=self.randomsearch_params['model'], 
                cv=self.randomsearch_params['cv'], 
                n_iter=self.randomsearch_params['n_iter'],
                n_jobs=self.randomsearch_params['n_jobs'], 
                scoring=make_scorer(matthews_corrcoef),
                verbose=self.verbose,
                refit = True
            )

            fitted_model = randomsearch.fit(X_train_rc, y_train_rc)

            print("Saving Results")
            result = pd.DataFrame(fitted_model.cv_results_)
            result['param_dwt_timewindow'] = dwt_timewindow
            result['param_examples_timewindow'] = examples_timewindow
            result['param_wavelet'] = wavelet

            if self.randomsearch_params['file_path']:
                results = open(self.randomsearch_params['file_path'], "a")
                writer = csv.writer(results, delimiter=",")
                for line in result.values:
                    writer.writerow(line)
                results.close()

    def generate_y(self, dfs, timestep, dwt_timewindow, dwt_overlap, examples_timewindow, examples_overlap):
        
        y = []
        examples_step = int((examples_timewindow - examples_overlap) / (dwt_timewindow- dwt_overlap))

        step = int((dwt_timewindow-dwt_overlap)/timestep)

        #Starting the conversion (goes through all the dataframes)
        for df in dfs:
            data = []
            
            current_index = step - 1

            while current_index < len(df):
                    
                data.append(1) if df.loc[df.index[current_index], self.column] > 20 else data.append(0)

                current_index += step
            
            current_index = examples_step - 1

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