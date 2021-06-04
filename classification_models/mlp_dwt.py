import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=RuntimeWarning)

from os.path import join, isfile
from os import listdir
import math
import json

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from scipy.stats import randint
from sklearn.metrics import matthews_corrcoef, confusion_matrix, make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import sys
import csv

sys.path.insert(1, "../feature_extractors")
from wt import get_discrete_features
from generate_timeseries import generate_appliance_timeseries, generate_main_timeseries
from tsfresh_extractor import get_tsfresh_features
from matthews_correlation import matthews_correlation

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

class MLP():
    def __init__(self, params):
        #Variable that will store the models trained for each appliance.
        self.model = {}
        #Name used to identify the Model Name.
        self.MODEL_NAME = params.get('model_name', 'MLP')
        #Percentage of values used as cross validation data from the training data.
        self.cv = params.get('cv', 0.16)
        #If this variable is not None than the class loads the appliance models present in the folder.
        self.load_model_path = params.get('load_model_folder',None)
        #Dictates the ammount of information to be presented during training and regression.
        self.verbose = params.get('verbose', 0)
        
        self.appliances = params.get('appliances', {})

        self.default_appliance = {
            "timewindow": 180,
            "overlap": 172,
            "timestep": 2,
            "feature_extractor": "wt",
            "wavelet": 'db4',
            "batch_size": 1024,
            "epochs": 300,
            "n_nodes":32
        }

        self.training_results_path = params.get("training_results_path", None)
        self.results_file = params.get("results_path", None)
        self.checkpoint_file = params.get("checkpoint_file", None)

        #In case of existing a model path, load every model in that path.
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
                overlap = appliance_model.get("overlap", self.default_appliance['overlap'])
                timestep = appliance_model.get("timestep", self.default_appliance['timestep'])
                batch_size = appliance_model.get("batch_size", self.default_appliance['batch_size'])
                epochs = appliance_model.get("epochs", self.default_appliance['epochs'])
                n_nodes = appliance_model.get("n_nodes", self.default_appliance['n_nodes'])
                feature_extractor = appliance_model.get("feature_extractor", self.default_appliance['feature_extractor'])

                if feature_extractor == "wt":
                    print("Using Discrete Wavelet Transforms as Features")
                    wavelet = appliance_model.get("wavelet", self.default_appliance['wavelet'])
                    X_train, self.mains_mean, self.mains_std = get_discrete_features(train_mains, wavelet, timestep, timewindow, overlap)
                elif feature_extractor == "tsfresh":
                    print("Using TSFresh as Features")
                    X_train, self.mains_mean, self.mains_std, self.parameters = get_tsfresh_features(train_mains, timestep, timewindow, overlap, app_dfs=appliance_power)
                else:
                    print("Using the Timeseries as Features")
                    X_train, self.mains_mean, self.mains_std = generate_main_timeseries(train_mains, timewindow, timestep, overlap)
                    X_train = X_train.reshape(X_train.shape[0], -1)
                
                y_train = generate_appliance_timeseries(appliance_power, True, timewindow, timestep, overlap)

                if self.verbose != 0 :
                    print("Nº de Positivos ", sum([ np.where(p == max(p))[0][0]  for p in y_train]))
                    print("Nº de Negativos ", y_train.shape[0]-sum([ np.where(p == max(p))[0][0]  for p in y_train ]))
                
                if self.verbose != 0:
                    print("Training ", app_name, " in ", self.MODEL_NAME, " model\n", end="\r")

                model = self.create_model(n_nodes, (X_train.shape[1],))

                checkpoint = ModelCheckpoint(self.checkpoint_file, monitor='val_loss', verbose=self.verbose, save_best_only=True, mode='min')

                history = model.fit(X_train, 
                        y_train,
                        epochs=epochs, 
                        batch_size=batch_size,
                        shuffle=True,
                        callbacks=[checkpoint],
                        validation_split=self.cv,
                        verbose=self.verbose
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
                appliance_model = self.appliances.get(app_name, {})
                timewindow = appliance_model.get("timewindow", self.default_appliance['timewindow'])
                timestep = appliance_model.get("timestep", self.default_appliance['timestep'])
                overlap = appliance_model.get("overlap", self.default_appliance['overlap'])
                feature_extractor = appliance_model.get("feature_extractor", self.default_appliance['feature_extractor'])

                if feature_extractor == "wt":
                    wavelet = appliance_model.get("wavelet", self.default_appliance['wavelet'])
                    X_train, self.mains_mean, self.mains_std = get_discrete_features(train_mains, wavelet, timestep, timewindow, overlap)
                elif feature_extractor == "tsfresh":
                    X_train, self.mains_mean, self.mains_std, self.parameters = get_tsfresh_features(train_mains, timestep, timewindow, overlap, app_dfs=appliance_power)
                else:
                    X_train, self.mains_mean, self.mains_std = generate_main_timeseries(train_mains, timewindow, timestep, overlap)

                print("Using Loaded Model")

    def disaggregate_chunk(self, test_mains, test_appliances):

        appliance_powers_dict = {}
        
        for app_name, appliance_power in test_appliances:
            if self.verbose != 0:
                print("Preparing the Test Data for %s" % app_name)

            appliance_model = self.appliances.get(app_name)
            timewindow = appliance_model.get("timewindow", self.default_appliance['timewindow'])
            overlap = appliance_model.get("overlap", self.default_appliance['overlap'])
            timestep = appliance_model.get("timestep", self.default_appliance['timestep'])
            feature_extractor = appliance_model.get("feature_extractor", self.default_appliance['feature_extractor'])
            
            if feature_extractor == "wt":
                print("Using Discrete Wavelet Transforms as Features")
                wavelet = appliance_model.get("wavelet", self.default_appliance['wavelet'])
                X_test = get_discrete_features(test_mains, wavelet, timestep, timewindow, overlap, self.mains_mean, self.mains_std)[0]
            elif feature_extractor == "tsfresh":
                print("Using TSFresh as Features")
                X_test = get_tsfresh_features(test_mains, timestep, timewindow, overlap, self.mains_mean, self.mains_std, parameters=self.parameters)[0]
            else:
                print("Using the Timeseries as Features")
                X_test = generate_main_timeseries(test_mains, timewindow, timestep, overlap, self.mains_mean, self.mains_std)[0]
                X_test = X_test.reshape(X_test.shape[0], -1)

            y_test = generate_appliance_timeseries(appliance_power, True, timewindow, timestep, overlap)
            
            if self.verbose != 0:
                print("Nº de Positivos ", sum([ np.where(p == max(p))[0][0]  for p in y_test ]))
                print("Nº de Negativos ", y_test.shape[0]-sum([ np.where(p == max(p))[0][0]  for p in y_test ]))

            if self.verbose != 0:
                print("Estimating power demand for '{}' in '{}'\n".format(app_name, self.MODEL_NAME))
            
            pred = self.model[app_name].predict(X_test)

            pred = [ np.where(p == max(p))[0][0]  for p in pred ]
            
            y_test = [ np.where(p == max(p))[0][0]  for p in y_test ]
            tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()

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
                f.write("MCC: "+str(matthews_corrcoef(y_test, pred))+ "\n")
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
        model.add(Dense(n_nodes, input_shape=input_shape, activation='relu'))
        model.add(Dense(int(n_nodes/4), activation='relu'))
        model.add(Dense(int(n_nodes/2), activation='relu'))
        model.add(Dense(int(n_nodes/4), activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(2, activation='softmax'))
        model.compile(optimizer=Adam(0.00001), loss='categorical_crossentropy', metrics=["accuracy", matthews_correlation])

        return model
