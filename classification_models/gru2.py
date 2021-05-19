import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=RuntimeWarning)

from os.path import join, isfile
from os import listdir
import json

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, GRU, LeakyReLU, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from scipy.stats import randint
from sklearn.metrics import matthews_corrcoef, confusion_matrix, make_scorer
from sklearn.model_selection import RandomizedSearchCV

import numpy as np

import sys
sys.path.insert(1, "../feature_extractors")
from generate_timeseries import generate_main_timeseries, generate_appliance_timeseries
from matthews_correlation import matthews_correlation

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

class GRU2():
    def __init__(self, params):
        #Variable that will store the models trained for each appliance.
        self.model = {}
        #Name used to identify the Model Name.
        self.MODEL_NAME = params.get('model_name', 'GRU2')
        #Percentage of values used as cross validation data from the training data.
        self.cv = params.get('cv', 0.16)
        #If this variable is not None than the class loads the appliance models present in the folder.
        self.load_model_path = params.get('load_model_folder',None)
        #Dictates the ammount of information to be presented during training and regression.
        self.verbose = params.get('verbose', 0)

        self.appliances = params.get('appliances', {})

        self.default_appliance = {
            "timewindow": 180,
            "overlap": 178,
            "timestep": 2,
            'epochs' : 1,
            'batch_size' : 1024,
            'n_nodes' : 90
        }

        self.training_results_path = params.get("training_results_path", None)
        self.checkpoint_file = params.get("checkpoint_file", None)
        self.results_file = params.get("results_path", None)

        if self.load_model_path:
            self.load_model(self.load_model_path)

    def partial_fit(self, train_mains, train_appliances):

        #For each appliance to be classified
        for app_name, appliance_power in train_appliances:
            if app_name not in self.model:
                if( self.verbose != 0):
                    print("Preparing Dataset for %s" % app_name)

                appliance_model = self.appliances.get(app_name, {})

                timewindow = appliance_model.get("timewindow", self.default_appliance['timewindow'])
                timestep = appliance_model.get("timestep", self.default_appliance['timestep'])
                overlap = appliance_model.get("overlap", self.default_appliance['overlap'])
                batch_size = appliance_model.get("batch_size", self.default_appliance['batch_size'])
                epochs = appliance_model.get("epochs", self.default_appliance['epochs'])
                n_nodes = appliance_model.get("n_nodes", self.default_appliance['n_nodes'])

                X_train, self.mains_mean, self.mains_std = generate_main_timeseries(train_mains, timewindow, timestep, overlap)  

                y_train = generate_appliance_timeseries(appliance_power, True, timewindow, timestep, overlap)

                if( self.verbose != 0):
                    print("Nº de Positivos ", sum([ np.where(p == max(p))[0][0]  for p in y_train]))
                    print("Nº de Negativos ", y_train.shape[0]-sum([ np.where(p == max(p))[0][0]  for p in y_train ]))

                if self.verbose != 0:
                    print("Training ", app_name, " in ", self.MODEL_NAME, " model\n", end="\r")
                
                model = self.create_model(n_nodes, (X_train.shape[1], X_train.shape[2]))

                checkpoint = ModelCheckpoint(self.checkpoint_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
                #Fits the model to the training data.
                history = model.fit( X_train, 
                        y_train, 
                        epochs=epochs, 
                        batch_size=batch_size, 
                        verbose=self.verbose, 
                        shuffle=True,
                        callbacks=[checkpoint],
                        validation_split=self.cv,
                        )
                history = json.dumps(history.history)

                if self.training_results_path is not None:
                    f = open(self.training_results_path + "history_"+app_name+"_"+self.MODEL_NAME+".json", "w")
                    f.write(history)
                    f.close()

                model.load_weights(self.checkpoint_file)
                
                #Stores the trained model.
                self.model[app_name] = model

                if self.results_file is not None:
                    f = open(self.results_file, "w")
                    f.write("Nº de Positivos para treino: " + str(sum([ np.where(p == max(p))[0][0]  for p in y_train])) + "\n")
                    f.write("Nº de Negativos para treino: " + str(y_train.shape[0]-sum([ np.where(p == max(p))[0][0]  for p in y_train ])) + "\n")
                    f.close()
            else:
                print("Using Loaded Model")
            
    def disaggregate_chunk(self, test_mains, test_appliances):
        
        appliance_powers_dict = {}
        
        for app_name, appliance_power in test_appliances:
            if self.verbose != 0:
                print("Preparing the Test Data for %s" % app_name)
            
            appliance_model = self.appliances.get(app_name, {})
            
            timewindow = appliance_model.get("timewindow", self.default_appliance['timewindow'])
            timestep = appliance_model.get("timestep", self.default_appliance['timestep'])
            overlap = appliance_model.get("overlap", self.default_appliance['overlap'])

            X_test = generate_main_timeseries(test_mains, timewindow, timestep, overlap, self.mains_mean, self.mains_std)[0] 

            y_test = generate_appliance_timeseries(appliance_power, True, timewindow, timestep, overlap)
            
            if( self.verbose != 0):
                print("Nº de Positivos ", sum([ np.where(p == max(p))[0][0]  for p in y_test ]))
                print("Nº de Negativos ", y_test.shape[0]-sum([ np.where(p == max(p))[0][0]  for p in y_test ]))

            if self.verbose != 0:
                print("Estimating power demand for '{}' in '{}'\n".format(app_name, self.MODEL_NAME))
            
            pred = self.model[app_name].predict(X_test)
            pred = [ np.where(p == max(p))[0][0]  for p in pred ]
        
            y_test = [ np.where(p == max(p))[0][0]  for p in y_test ]
            
            tn, fn, fp, tp = confusion_matrix(y_test, pred).ravel()

            if self.verbose == 2:
                print("True Positives: ", tp)
                print("True Negatives: ", tn)  
                print("False Negatives: ", fn)  
                print("False Positives: ", fp)        
                print( "MCC: ", matthews_corrcoef(y_test, pred))

            if self.results_file is not None:
                f = open(self.results_file, "a")
                f.write("Nº de Positivos para teste: " + str(sum(y_test)) + "\n")
                f.write("Nº de Negativos para teste: " + str(len(y_test)- sum(y_test)) + "\n")
                f.write("MCC: "+str(matthews_corrcoef(y_test, pred)))
                f.write("True Positives: "+str(tp)+ "\n")
                f.write("True Negatives: "+str(tn)+ "\n")
                f.write("False Positives: "+str(fp)+ "\n")
                f.write("False Negatives: "+str(fn)+ "\n")
                f.close()

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
            self.model[app.split(".")[0].split("_")[2]] = load_model(join(folder_name, app), custom_objects={"matthews_correlation":matthews_correlation})

    def create_model(self, n_nodes, input_shape):
        #Creates a specific model.
        model = Sequential()
        model.add(GRU(n_nodes, input_shape=input_shape, return_sequences=True, activation='relu'))
        model.add(GRU(n_nodes*2, return_sequences=True, activation='relu'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(GRU(n_nodes, return_sequences=True, activation='relu'))
        model.add(GRU(int(n_nodes/2), activation='relu'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dense(int(n_nodes/4), activation='relu'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dropout(0.1))
        model.add(Dense(2, activation='softmax'))
        model.compile(optimizer=Adam(0.00001), loss='categorical_crossentropy', metrics=["accuracy", matthews_correlation])

        return model