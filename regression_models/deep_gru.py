
from os.path import join, isfile
from os import listdir
import json
import math
import random

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GRU, LeakyReLU, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.metrics import mean_squared_error, mean_absolute_error

import pandas as pd
import numpy as np

#import sys
#sys.path.insert(1, "../utils")
#sys.path.insert(1, "../feature_extractors")

from generate_timeseries import generate_main_timeseries, generate_appliance_timeseries

import utils
import plots

class DeepGRU():
    def __init__(self, params):
        #Variable that will store the models trained for each appliance.
        self.model = {}
        #Name used to identify the Model Name.
        self.MODEL_NAME = params.get('model_name', 'DeepGRU')
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
            "overlap": 178,
            "timestep": 2,
            'epochs' : 1,
            'batch_size' : 1024,
            'n_nodes' : 90,
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

        if self.load_model_path:
            self.load_model(self.load_model_path)
            self.mains_mean = params.get("mean", None)
            self.mains_std = params.get("std", None)

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
                app_mean = appliance_model.get("mean", None)
                app_std = appliance_model.get("std", None)
                on_treshold = appliance_model.get("on_treshold", self.default_appliance['on_treshold'])

                if self.mains_mean is None:
                    X_train, self.mains_mean, self.mains_std = generate_main_timeseries(train_mains[0:-1], timewindow, timestep, overlap)
                    X_cv = generate_main_timeseries([train_mains[-1]], timewindow, timestep, overlap, self.mains_mean, self.mains_std)[0]
                else:
                    X_train = generate_main_timeseries(train_mains[0:-1], timewindow, timestep, overlap, self.mains_mean, self.mains_std)[0]  
                    X_cv = generate_main_timeseries([train_mains[-1]], timewindow, timestep, overlap, self.mains_mean, self.mains_std)[0]

                if app_mean is None:
                    y_train, app_mean, app_std = generate_appliance_timeseries(appliance_power[0:-1], False, timewindow, timestep, overlap)
                    y_cv = generate_appliance_timeseries([appliance_power[-1]], False, timewindow, timestep, overlap, app_mean, app_std)[0]
                    appliance_model["mean"] = app_mean
                    appliance_model["std"] = app_std
                else:
                    y_train = generate_appliance_timeseries(appliance_power[0:-1], False, timewindow, timestep, overlap, app_mean, app_std)[0]
                    y_cv = generate_appliance_timeseries([appliance_power[-1]], False, timewindow, timestep, overlap, app_mean, app_std)[0]

                binary_y = np.array([ 1 if x > on_treshold else 0 for x in (y_train*app_std) + app_mean])
                
                negatives = np.where(binary_y == 0)[0]
                positives = np.where(binary_y == 1)[0]

                negatives = list(random.sample(set(negatives), positives.shape[0]))
                undersampled_dataset = np.sort(np.concatenate((positives, negatives)))

                X_train = X_train[undersampled_dataset]
                y_train = y_train[undersampled_dataset]

                n_activatons = positives.shape[0]
                on_examples = n_activatons / y_train.shape[0]
                off_examples = (y_train.shape[0] - n_activatons) / y_train.shape[0]

                binary_y = np.array([ 1 if x > on_treshold else 0 for x in (y_cv*app_std) + app_mean])
                
                negatives = np.where(binary_y == 0)[0]
                positives = np.where(binary_y == 1)[0]

                negatives = list(random.sample(set(negatives), positives.shape[0]))
                undersampled_dataset = np.sort(np.concatenate((positives, negatives)))

                #X_cv = X_cv[undersampled_dataset]
                #y_cv = y_cv[undersampled_dataset]

                
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
                
                model = self.create_model(n_nodes, (X_train.shape[1], X_train.shape[2]))

                checkpoint = ModelCheckpoint(
                                self.checkpoint_folder + "model_checkpoint_" + app_name.replace(" ", "_") + ".h5", 
                                monitor='val_loss', 
                                verbose=1, 
                                save_best_only=True, 
                                mode='min'
                                )

                #Fits the model to the training data.
                history = model.fit( X_train, 
                        y_train, 
                        epochs=epochs, 
                        batch_size=batch_size, 
                        verbose=1, 
                        shuffle=False,
                        callbacks=[checkpoint],
                        validation_data=(X_cv, y_cv),
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

                train_rmse = math.sqrt(mean_squared_error(y_train* app_std + app_mean, pred))
                train_mae = mean_absolute_error(y_train* app_std + app_mean, pred)

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
            
            appliance_model = self.appliances.get(app_name, {})
            
            timewindow = appliance_model.get("timewindow", self.default_appliance['timewindow'])
            timestep = appliance_model.get("timestep", self.default_appliance['timestep'])
            overlap = appliance_model.get("overlap", self.default_appliance['overlap'])
            app_mean = appliance_model.get("mean", None)
            app_std = appliance_model.get("std", None)

            X_test = generate_main_timeseries(test_mains, timewindow, timestep, overlap, self.mains_mean, self.mains_std)[0] 

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
            self.model[app].save(join(folder_name, app + ".h5"))

    def load_model(self, folder_name):
        #Get all the models trained in the given folder and load them.
        app_models = [f for f in listdir(folder_name) if isfile(join(folder_name, f))]
        for app in app_models:
            self.model[app.split(".")[0]] = load_model(join(folder_name, app))

    def create_model(self, n_nodes, input_shape):
        input = Input(input_shape)
        #Block 1
        gru1 = GRU(n_nodes, return_sequences=True)(input)
        #Block 2
        gru2 = GRU(n_nodes*2, return_sequences=True)(gru1)
        #Block 3
        gru3 = GRU(n_nodes*2, return_sequences=True)(gru2)
        #Block 4
        gru4 = GRU(n_nodes*2)(gru3)

        #Dense Layers Voltar a adicionar(faz sentido)
        dense1 = Dense(n_nodes*2, activation='relu')(gru4)
        
        #Classification Layer
        output = Dense(1)(dense1)

        model = Model(inputs=input, outputs=output)
        model.compile(loss='mean_squared_error', metrics=["MeanAbsoluteError", "RootMeanSquaredError"], optimizer='adam')

        return model

    def create_transfer_model(self, transfer_path, input_shape):
        trained_model = load_model(transfer_path)
        trained_model.layers.pop(0)
        trained_model.layers.pop(-1)
        for layer in trained_model.layers:
            layer.trainable = False

        new_input = Input(input_shape)
        freezed_layers = trained_model(new_input)
        new_output = Dense(1)(freezed_layers)

        model = Model(inputs=new_input, outputs=new_output)
        
        model.compile(loss='mean_squared_error', metrics=["MeanAbsoluteError", "RootMeanSquaredError"], optimizer='adam')

        return model
