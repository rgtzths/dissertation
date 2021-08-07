
from os.path import join, isfile
from os import listdir
import json
import math

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, GRU, Dropout, InputLayer
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
        self.cv_split = params.get('cv_split', 0.16)
        #If this variable is not None than the class loads the appliance models present in the folder.
        self.load_model_path = params.get('pretrained-model-path',None)
        #Dictates the ammount of information to be presented during training and regression.
        self.verbose = params.get('verbose', 0)

        self.mains_mean = params.get("mains_mean", None)
        self.mains_std = params.get("mains_std", None)

        self.appliances = params["appliances"]

        self.default_appliance = {
            "timewindow": 180,
            "overlap": 178,
            'epochs' : 300,
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

    def partial_fit(self, train_data, cv_data=None):

        #For each appliance to be classified
        for app_name, data in train_data.items():
            if( self.verbose != 0):
                print("Preparing Dataset for %s" % app_name)

            appliance_model = self.appliances[app_name]

            timewindow = appliance_model.get("timewindow", self.default_appliance['timewindow'])
            timestep = appliance_model["timestep"]
            overlap = appliance_model.get("overlap", self.default_appliance['overlap'])
            batch_size = appliance_model.get("batch_size", self.default_appliance['batch_size'])
            epochs = appliance_model.get("epochs", self.default_appliance['epochs'])
            n_nodes = appliance_model.get("n_nodes", self.default_appliance['n_nodes'])
            app_mean = appliance_model.get("mean", None)
            app_std = appliance_model.get("std", None)
            on_treshold = appliance_model.get("on_treshold", self.default_appliance['on_treshold'])
            transfer_path = appliance_model.get("transfer_path", None)

            if self.mains_mean is None:
                X_train, self.mains_mean, self.mains_std = generate_main_timeseries(data["mains"], timewindow, timestep, overlap)
            else:
                X_train = generate_main_timeseries(data["mains"], timewindow, timestep, overlap, self.mains_mean, self.mains_std)[0]  

            if app_mean is None:
                y_train, app_mean, app_std = generate_appliance_timeseries(data["appliance"], False, timewindow, timestep, overlap)
                appliance_model["mean"] = app_mean
                appliance_model["std"] = app_std
            else:
                y_train = generate_appliance_timeseries(data["appliance"], False, timewindow, timestep, overlap, app_mean, app_std)[0]

            binary_y = np.array([ 1 if x > on_treshold else 0 for x in (y_train*app_std) + app_mean])
            
            train_positives = np.where(binary_y == 1)[0]

            train_n_activatons = train_positives.shape[0]
            train_on_examples = train_n_activatons / y_train.shape[0]
            train_off_examples = (y_train.shape[0] - train_n_activatons) / y_train.shape[0]
            
            if cv_data is not None:
                X_cv = generate_main_timeseries(cv_data[app_name]["mains"], timewindow, timestep, overlap, self.mains_mean, self.mains_std)[0]
                y_cv = generate_appliance_timeseries(cv_data[app_name]["appliance"], False, timewindow, timestep, overlap, app_mean, app_std)[0]
            
                binary_y = np.array([ 1 if x > on_treshold else 0 for x in (y_cv*app_std) + app_mean])
                
                cv_positives = np.where(binary_y == 1)[0]

                cv_n_activatons = cv_positives.shape[0]
                cv_on_examples = cv_n_activatons / y_cv.shape[0]
                cv_off_examples = (y_cv.shape[0] - cv_n_activatons) / y_cv.shape[0]

            if( self.verbose == 2):
                print("-"*5 + "Train Info" + "-"*5)
                print("Nº of examples: ", str(X_train.shape[0]))
                print("Nº of activations: ", str(train_n_activatons))
                print("On Percentage: ", str(train_on_examples))
                print("Off Percentage: ", str(train_off_examples))
                if cv_data is not None:
                    print("-"*5 + "Cross Validation Info" + "-"*5)
                    print("Nº of examples: ", str(X_cv.shape[0]))
                    print("Nº of activations: ", str(cv_n_activatons))
                    print("On Percentage: ", str(cv_on_examples))
                    print("Off Percentage: ", str(cv_off_examples))
                print("-"*10)
                print("Mains Mean: ", str(self.mains_mean))
                print("Mains Std: ", str(self.mains_std))
                print(app_name + " Mean: ", str(app_mean))
                print(app_name + " Std: ", str(app_std))
            
            if app_name in self.model:
                if self.verbose > 0:
                    print("Starting from previous step")
                model = self.model[app_name]
            else:
                if transfer_path is None:
                    if self.verbose > 0:
                        print("Creating new model")
                    model = self.create_model(n_nodes, (X_train.shape[1], X_train.shape[2]))       
                else:
                    if self.verbose > 0:
                        print("Starting from pre-trained model")
                    model = self.create_transfer_model(transfer_path, (X_train.shape[1], X_train.shape[2]), n_nodes)

            if self.verbose != 0:
                print("Training ", app_name, " in ", self.MODEL_NAME, " model\n", end="\r")
            
            verbose = 1 if self.verbose >= 1 else 0

            checkpoint = ModelCheckpoint(
                            self.checkpoint_folder + "model_checkpoint_" + app_name.replace(" ", "_") + ".h5", 
                            monitor='val_loss', 
                            verbose=verbose, 
                            save_best_only=True, 
                            mode='min'
                            )

            #Fits the model to the training data.
            if cv_data is not None:
                history = model.fit(X_train, 
                        y_train,
                        epochs=epochs, 
                        batch_size=batch_size,
                        shuffle=False,
                        callbacks=[checkpoint],
                        validation_data=(X_cv, y_cv),
                        verbose=verbose
                        )        
            else:
                history = model.fit(X_train, 
                    y_train,
                    epochs=epochs, 
                    batch_size=batch_size,
                    shuffle=False,
                    callbacks=[checkpoint],
                    validation_split=self.cv_split,
                    verbose=verbose
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
            #Concatenates training and cross_validation
            X = np.concatenate((X_train, X_cv), axis=0)
            y = np.concatenate((y_train, y_cv), axis=0)

            pred = self.model[app_name].predict(X) * app_std + app_mean

            train_rmse = math.sqrt(mean_squared_error(y * app_std + app_mean, pred))
            train_mae = mean_absolute_error(y * app_std + app_mean, pred)

            if self.verbose == 2:
                print("Training scores")    
                print("RMSE: ", train_rmse )
                print("MAE: ", train_mae )
            
            if self.results_folder is not None:
                f = open(self.results_folder + "results_" + app_name.replace(" ", "_") + ".txt", "w")
                f.write("-"*5 + "Train Info" + "-"*5)
                f.write("Nº of examples: "+ str(X_train.shape[0]))
                f.write("Nº of activations: "+ str(train_n_activatons))
                f.write("On Percentage: "+ str(train_on_examples))
                f.write("Off Percentage: "+ str(train_off_examples))
                if cv_data is not None:
                    f.write("-"*5 + "Cross Validation Info" + "-"*5)
                    f.write("Nº of examples: "+ str(X_cv.shape[0]))
                    f.write("Nº of activations: "+ str(cv_n_activatons))
                    f.write("On Percentage: "+ str(cv_on_examples))
                    f.write("Off Percentage: "+ str(cv_off_examples))
                f.write("-"*10)
                f.write("Mains Mean: " + str(self.mains_mean) + "\n")
                f.write("Mains Std: " + str(self.mains_std) + "\n")
                f.write(app_name + " Mean: " + str(app_mean) + "\n")
                f.write(app_name + " Std: " + str(app_std) + "\n")
                f.write("Train RMSE: "+str(train_rmse)+ "\n")
                f.write("Train MAE: "+str(train_mae)+ "\n")
                f.close()

            
    def disaggregate_chunk(self, test_mains, app_name):

        test_predictions_list = []
        appliance_powers_dict = {}

        if self.verbose != 0:
            print("Preparing the Test Data for %s" % app_name)
        
        appliance_model = self.appliances.get(app_name, {})
        
        timewindow = appliance_model.get("timewindow", self.default_appliance['timewindow'])
        timestep = appliance_model["timestep"]
        overlap = appliance_model.get("overlap", self.default_appliance['overlap'])
        app_mean = appliance_model.get("mean", None)
        app_std = appliance_model.get("std", None)

        X_test = generate_main_timeseries(test_mains, timewindow, timestep, overlap, self.mains_mean, self.mains_std)[0] 

        if( self.verbose == 2):
            print("Nº of test examples", X_test.shape[0])

        if self.verbose != 0:
            print("Estimating power demand for '{}' in '{}'\n".format(app_name, self.MODEL_NAME))
        
        pred = self.model[app_name].predict(X_test).flatten()* app_std + app_mean
        pred = np.where(pred > 0, pred, 0)

        appliance_powers_dict[app_name] = pd.Series(pred)
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
        model = Sequential()
        model.add(InputLayer(input_shape))
        model.add(GRU(n_nodes, return_sequences=True))
        model.add(GRU(n_nodes*2, return_sequences=True))
        model.add(GRU(n_nodes*2, return_sequences=True))
        model.add(GRU(n_nodes*2))
        model.add(Dense(n_nodes*2, activation='relu'))
        #model.add(Dropout(0.5))
        model.add(Dense(1))

        model.compile(loss='mean_squared_error', metrics=["MeanAbsoluteError", "RootMeanSquaredError"], optimizer='adam')

        return model

    def create_transfer_model(self, transfer_path, input_shape, n_nodes=90):
        model = Sequential()
        model.add(InputLayer(input_shape))
        model.add(GRU(n_nodes, return_sequences=True))
        model.add(GRU(n_nodes*2, return_sequences=True))
        model.add(GRU(n_nodes*2, return_sequences=True))
        model.add(GRU(n_nodes))
        
        model.load_weights(transfer_path, skip_mismatch=True, by_name=True)

        for layer in model.layers:
            layer.trainable = False
        
        model.add(Dense(int(n_nodes*2), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))

        model.compile(loss='mean_squared_error', metrics=["MeanAbsoluteError", "RootMeanSquaredError"], optimizer='adam')

        return model
