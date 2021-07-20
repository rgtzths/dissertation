
from os.path import join, isfile
from os import listdir
import math
import json
import random

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

#import sys
#sys.path.insert(1, "../utils")
#sys.path.insert(1, "../feature_extractors")

from wt import get_discrete_features
from generate_timeseries import generate_appliance_timeseries, generate_main_timeseries

import utils
import plots

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

        self.mains_mean = params.get("mains_mean", None)
        self.mains_std = params.get("mains_std", None)
        
        self.appliances = params.get('appliances', {})

        self.default_appliance = {
            "timewindow": 180,
            "overlap": 172,
            "timestep": 2,
            "feature_extractor": "",
            "wavelet": 'db4',
            "batch_size": 1024,
            "epochs": 1,
            "n_nodes":256,
            "on_treshold" : 50
        }

        self.training_history_folder = params.get("training_history_folder", None)
        self.results_folder = params.get("results_folder", None)
        self.checkpoint_folder = params.get("checkpoint_folder", None)
        self.plots_folder = params.get("plots_folder", None)

        if self.training_history_folder is not None:
            utils.create_path(self.training_history_folder)
        
        if self.results_folder is not None:
            utils.create_path(self.results_folder)
        
        if self.checkpoint_folder is not None:
            utils.create_path(self.checkpoint_folder)
        
        if self.plots_folder is not None:
            utils.create_path(self.plots_folder)

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
                app_mean = appliance_model.get("mean", None)
                app_std = appliance_model.get("std", None)
                on_treshold = appliance_model.get("on_treshold", self.default_appliance['on_treshold'])

                if feature_extractor == "wt":
                    print("Using Discrete Wavelet Transforms as Features")
                    wavelet = appliance_model.get("wavelet", self.default_appliance['wavelet'])
                    if self.mains_mean is None:
                        X_train, self.mains_mean, self.mains_std = get_discrete_features(train_mains, wavelet, timestep, timewindow, overlap)
                    else:
                        X_train = get_discrete_features(train_mains, wavelet, timestep, timewindow, overlap, self.mains_mean, self.mains_std)[0]
                else:
                    print("Using the Timeseries as Features")
                    if self.mains_mean is None:
                        X_train, self.mains_mean, self.mains_std = generate_main_timeseries(train_mains, timewindow, timestep, overlap)
                    else:
                        X_train = generate_main_timeseries(train_mains, timewindow, timestep, overlap, self.mains_mean, self.mains_std)[0]
                    X_train = X_train.reshape(X_train.shape[0], -1)
                
                if app_mean is None:
                    y_train, app_mean, app_std = generate_appliance_timeseries(appliance_power, False, timewindow, timestep, overlap)
                    appliance_model["mean"] = app_mean
                    appliance_model["std"] = app_std
                else:
                    y_train = generate_appliance_timeseries(appliance_power, False, timewindow, timestep, overlap, app_mean, app_std)[0]

                binary_y = np.array([ 1 if x > on_treshold else 0 for x in (y_train*app_std) + app_mean])
                
                negatives = np.where(binary_y == 0)[0]
                positives = np.where(binary_y == 1)[0]

                negatives = list(random.sample(set(negatives), positives.shape[0]))
                undersampled_dataset = np.sort(np.concatenate((positives, negatives)))

                X_train = X_train[undersampled_dataset]
                y_train = y_train[undersampled_dataset]

                binary_y = [ 1 if x > on_treshold else 0 for x in (y_train*app_std) + app_mean]
                n_activatons = sum(binary_y)
                on_examples = n_activatons / len(binary_y)
                off_examples = (len(binary_y) - n_activatons) / len(binary_y)

                if( self.verbose != 0):
                    print("Nº of examples ", str(X_train.shape[0]))
                    print("Nº of activations for training: ", str(n_activatons))
                    print("On Percentage: ", str(on_examples))
                    print("Off Percentage: ", str(off_examples))
                    print("Mains Mean: ", str(self.mains_mean))
                    print("Mains Std: ", str(self.mains_std))
                    print(app_name + " Mean: ", str(app_mean))
                    print(app_name + " Std: ", str(app_std))
                
                if self.verbose != 0:
                    print("Training ", app_name, " in ", self.MODEL_NAME, " model\n", end="\r")

                model = self.create_model(n_nodes, (X_train.shape[1],))

                checkpoint = ModelCheckpoint(
                                self.checkpoint_folder + "model_checkpoint_" + app_name.replace(" ", "_") + ".h5", 
                                monitor='val_loss', 
                                verbose=self.verbose, 
                                save_best_only=True, 
                                mode='min'
                                )

                history = model.fit(X_train, 
                        y_train,
                        epochs=epochs, 
                        batch_size=batch_size,
                        shuffle=False,
                        callbacks=[checkpoint],
                        validation_split=self.cv,
                        verbose=self.verbose
                        )         

                history = json.dumps(history.history)

                if self.training_history_folder is not None:
                    f = open(self.training_history_folder + "history_"+app_name.replace(" ", "_")+".json", "w")
                    f.write(history)
                    f.close()
                
                if self.plots_folder is not None:
                    utils.create_path(self.plots_folder + "/" + app_name + "/")
                    plots.plot_model_history_regression(json.loads(history), self.plots_folder + "/" + app_name + "/")

                model.load_weights(self.checkpoint_folder + "model_checkpoint_" + app_name.replace(" ", "_") + ".h5")

                #Stores the trained model.
                self.model[app_name] = model
                
                #Gets the trainning data score
                pred = self.model[app_name].predict(X_train) * app_std + app_mean

                train_rmse = math.sqrt(mean_squared_error(y_train, pred))
                train_mae = mean_absolute_error(y_train, pred)

                if self.verbose == 2:
                    print("Training scores")    
                    print("RMSE: ", train_rmse )
                    print("MAE: ", train_mae )
                
                if self.results_folder is not None:
                    f = open(self.results_folder + "results_" + app_name.replace(" ", "_") + ".txt", "w")
                    f.write("Nº of examples for training: " + str(y_train.shape[0]) + "\n")
                    f.write("Nº of activations for training: " + str(n_activatons) + "\n")
                    f.write("On Percentage: " + str(on_examples) + "\n")
                    f.write("Off Percentage: " + str(off_examples) + "\n")
                    f.write("Mains Mean: " + str(self.mains_mean) + "\n")
                    f.write("Mains Std: " + str(self.mains_std) + "\n")
                    f.write(app_name + " Mean: " + str(app_mean) + "\n")
                    f.write(app_name + " Std: " + str(app_std) + "\n")
                    f.write("Train RMSE: "+str(train_rmse)+ "\n")
                    f.write("Train MAE: "+str(train_mae)+ "\n")
                    f.close()
            else:
                print("Using Loaded Model")

    def disaggregate_chunk(self, test_mains):

        test_predictions_list = []

        appliance_powers_dict = {}
        
        for i, app_name in enumerate(self.model):

            if self.verbose != 0:
                print("Preparing the Test Data for %s" % app_name)

            appliance_model = self.appliances.get(app_name)
            timewindow = appliance_model.get("timewindow", self.default_appliance['timewindow'])
            overlap = appliance_model.get("overlap", self.default_appliance['overlap'])
            timestep = appliance_model.get("timestep", self.default_appliance['timestep'])
            feature_extractor = appliance_model.get("feature_extractor", self.default_appliance['feature_extractor'])
            app_mean = appliance_model.get("mean", None)
            app_std = appliance_model.get("std", None)
            
            if feature_extractor == "wt":
                print("Using Discrete Wavelet Transforms as Features")
                wavelet = appliance_model.get("wavelet", self.default_appliance['wavelet'])
                X_test = get_discrete_features(test_mains, wavelet, timestep, timewindow, overlap, self.mains_mean, self.mains_std)[0]
            else:
                print("Using the Timeseries as Features")
                X_test = generate_main_timeseries(test_mains, timewindow, timestep, overlap, self.mains_mean, self.mains_std)[0]
                X_test = X_test.reshape(X_test.shape[0], -1)
            
            if( self.verbose != 0):
                print("Nº of examples", X_test.shape[0])

            if self.verbose != 0:
                print("Estimating power demand for '{}' in '{}'\n".format(app_name, self.MODEL_NAME))
            
            pred = self.model[app_name].predict(X_test)* app_std + app_mean

            pred = [p[0] for p in pred]

            column = pd.Series(
                    pred, index=test_mains[0].index, name=i)
            appliance_powers_dict[app_name] = column
        
        test_predictions_list.append(pd.DataFrame(appliance_powers_dict, dtype='float32'))

        return test_predictions_list

    def save_model(self, folder_name):
        #For each appliance trained store its model
        for app in self.model:
            self.model[app].save(join(folder_name, app + "_" + self.MODEL_NAME + ".h5"))

    def load_model(self, folder_name):
        #Get all the models trained in the given folder and load them.
        app_models = [f for f in listdir(folder_name) if isfile(join(folder_name, f))]
        for app in app_models:
            self.model[app.split(".")[0].split("_")[2]] = load_model(join(folder_name, app))

    def create_model(self, n_nodes, input_shape):
        #Creates a specific model.
        model = Sequential()
        model.add(Dense(n_nodes, input_shape=input_shape, activation='relu'))
        model.add(Dense(int(n_nodes/2), activation='relu'))
        model.add(Dense(int(n_nodes/4), activation='relu'))
        model.add(Dense(int(n_nodes/4), activation='relu'))
        model.add(Dropout(0.1))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer=Adam(0.00001), metrics=["MeanAbsoluteError", "RootMeanSquaredError"])

        return model
