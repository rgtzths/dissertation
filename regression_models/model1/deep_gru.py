
from os.path import join, isfile
from os import listdir
import json
import math

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, GRU, Dropout, InputLayer, Flatten, LeakyReLU
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from kerastuner.tuners import RandomSearch

from sklearn.metrics import mean_squared_error, mean_absolute_error

import pandas as pd
import numpy as np
from tensorflow.python.keras.utils.generic_utils import default

#import sys
#sys.path.insert(1, "../utils")
#sys.path.insert(1, "../feature_extractors")

from generate_timeseries import generate_main_timeseries, generate_appliance_timeseries

import utils
import plots

def model_builder(hp):
    model = Sequential()
    model.add(InputLayer((299,1)))
    hp_units = hp.Int('gru_1', min_value = 32, max_value = 256, step = 32, default=128)
    model.add(GRU(hp_units, return_sequences=True))

    hp_units = hp.Int('dense_1', min_value = 32, max_value = 256, step = 32, default=128)
    model.add(Dense(hp_units, activation='relu'))

    hp_units = hp.Float('drop_1', min_value = 0, max_value = 0.5, step = 0.05, default=0.2)
    model.add(Dropout(hp_units))

    hp_units = hp.Int('gru_2', min_value = 32, max_value = 256, step = 32, default=64)
    model.add(GRU(hp_units, return_sequences=True))

    hp_units = hp.Int('dense_2', min_value = 32, max_value = 256, step = 32, default=64)
    model.add(Dense(hp_units, activation='relu'))

    hp_units = hp.Float('drop_2', min_value = 0, max_value = 0.5, step = 0.05, default=0.2)
    model.add(Dropout(hp_units))

    hp_units = hp.Int('gru_3', min_value = 32, max_value = 256, step = 32, default=32)
    model.add(GRU(hp_units))

    hp_units = hp.Int('dense_3', min_value = 32, max_value = 256, step = 32, default=32)
    model.add(Dense(hp_units, activation='relu'))

    hp_units = hp.Float('drop_3', min_value = 0, max_value = 0.5, step = 0.05, default=0.2)
    model.add(Dropout(hp_units))

    model.add(Dense(1))

    model.compile(loss='mean_squared_error', metrics=["MeanAbsoluteError", "RootMeanSquaredError"], optimizer='adam')

    return model


class DeepGRU():
    def __init__(self, params):
        #Variable that will store the models trained for each appliance.
        self.model = {}
        #Name used to identify the Model Name.
        self.MODEL_NAME = params.get('model_name', 'DeepGRU')
        #Percentage of values used as cross validation data from the training data.
        self.cv_split = params.get('cv_split', 0.16)
        #If this variable is not None than the class loads the appliance models present in the folder.
        self.load_model_path = params.get('load_model_path',None)
        #Dictates the ammount of information to be presented during training and regression.
        self.verbose = params.get('verbose', 1)

        self.appliances = params["appliances"]

        self.random_search = params.get("random_search", False)

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
            mains_std = appliance_model.get("mains_std", None)
            mains_mean = appliance_model.get("mains_mean", None)

            if mains_mean is None:
                X_train, mains_mean, mains_std = generate_main_timeseries(data["mains"], timewindow, timestep, overlap)
                appliance_model["mains_mean"] = mains_mean
                appliance_model["mains_std"] = mains_std
            else:
                X_train = generate_main_timeseries(data["mains"], timewindow, timestep, overlap, mains_mean, mains_std)[0]  

            if app_mean is None:
                y_train, app_mean, app_std = generate_appliance_timeseries(data["appliance"], False, timewindow, timestep, overlap)
                appliance_model["mean"] = app_mean
                appliance_model["std"] = app_std
            else:
                y_train = generate_appliance_timeseries(data["appliance"], False, timewindow, timestep, overlap, app_mean=app_mean, app_std=app_std)[0]

            binary_y = np.array([ 1 if x > on_treshold else 0 for x in (y_train*app_std) + app_mean])
            
            train_positives = np.where(binary_y == 1)[0]

            train_n_activatons = train_positives.shape[0]
            train_on_examples = train_n_activatons / y_train.shape[0]
            train_off_examples = (y_train.shape[0] - train_n_activatons) / y_train.shape[0]
            
            if cv_data is not None:
                X_cv = generate_main_timeseries(cv_data[app_name]["mains"], timewindow, timestep, overlap, mains_mean, mains_std)[0]
                y_cv = generate_appliance_timeseries(cv_data[app_name]["appliance"], False, timewindow, timestep, overlap, app_mean=app_mean, app_std=app_std)[0]
            
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
                print("Mains Mean: ", str(mains_mean))
                print("Mains Std: ", str(mains_std))
                print(app_name + " Mean: ", str(app_mean))
                print(app_name + " Std: ", str(app_std))
            
            if self.random_search:
                tuner = RandomSearch(
                    model_builder,
                    max_trials=30,
                    objective='val_loss',
                    seed=10,
                    executions_per_trial=1,
                    overwrite=True,
                    directory='random_search_deep_gru',
                    project_name='nilm'
                )

                tuner.search(X_train, y_train, epochs=150, validation_data=(X_cv, y_cv))
                
                # Show a summary of the search
                tuner.results_summary()

                if cv_data is not None:
                    X = np.concatenate((X_train, X_cv), axis=0)
                    y = np.concatenate((y_train, y_cv), axis=0)
                else:
                    X = X_train
                    y = y_train

                # Retrieve the best model.
                model = tuner.get_best_models(num_models=1)[0]
            
            else:
                if app_name in self.model:
                    if self.verbose > 0:
                        print("Starting from previous step")
                    model = self.model[app_name]
                else:
                    if transfer_path is None:
                        if self.verbose > 0:
                            print("Creating new model")
                        model = self.create_model(90, (X_train.shape[1], X_train.shape[2]))       
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
                

            #Gets the trainning data score
            #Concatenates training and cross_validation
            if cv_data is not None:
                X = np.concatenate((X_train, X_cv), axis=0)
                y = np.concatenate((y_train, y_cv), axis=0)
            else:
                X = X_train
                y = y_train


            if transfer_path is not None:
                model.summary()
                for layer in model.layers:
                    layer.trainable = True

                model.compile(loss='mean_squared_error', metrics=["MeanAbsoluteError", "RootMeanSquaredError"], optimizer=Adam(1e-5))
                model.summary()
                model.fit(X, 
                        y,
                        epochs=1, 
                        batch_size=batch_size,
                        shuffle=False,
                        verbose=verbose
                        )

            #Stores the trained model.
            self.model[app_name] = model

            pred = self.model[app_name].predict(X) * app_std + app_mean

            train_rmse = math.sqrt(mean_squared_error(y * app_std + app_mean, pred))
            train_mae = mean_absolute_error(y * app_std + app_mean, pred)

            if self.verbose == 2:
                print("Training scores")    
                print("RMSE: ", train_rmse )
                print("MAE: ", train_mae )
            
            if self.results_folder is not None:
                f = open(self.results_folder + "results_" + app_name.replace(" ", "_") + ".txt", "w")
                f.write("-"*5 + "Train Info" + "-"*5+ "\n")
                f.write("Nº of examples: "+ str(X_train.shape[0])+ "\n")
                f.write("Nº of activations: "+ str(train_n_activatons)+ "\n")
                f.write("On Percentage: "+ str(train_on_examples)+ "\n")
                f.write("Off Percentage: "+ str(train_off_examples)+ "\n")
                if cv_data is not None:
                    f.write("-"*5 + "Cross Validation Info" + "-"*5+ "\n")
                    f.write("Nº of examples: "+ str(X_cv.shape[0])+ "\n")
                    f.write("Nº of activations: "+ str(cv_n_activatons)+ "\n")
                    f.write("On Percentage: "+ str(cv_on_examples)+ "\n")
                    f.write("Off Percentage: "+ str(cv_off_examples)+ "\n")
                f.write("-"*10+ "\n")
                if self.random_search:
                    best_hp = tuner.get_best_hyperparameters()[0]
                    f.write("-"*5 + "Best Hyperparameters" + "-"*5+ "\n")
                    print(best_hp.values)
                    f.write(str(best_hp.values)+"\n")
                    f.write("-"*10+ "\n")
                f.write("Mains Mean: " + str(mains_mean) + "\n")
                f.write("Mains Std: " + str(mains_std) + "\n")
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
        app_mean = appliance_model["mean"]
        app_std = appliance_model["std"]
        mains_std = appliance_model["mains_std"]
        mains_mean = appliance_model["mains_mean"]

        X_test = generate_main_timeseries(test_mains, timewindow, timestep, overlap, mains_mean, mains_std)[0] 

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
            self.model[app].save(join(folder_name, app.replace(" ", "_")+ ".h5"))

            app_params = self.appliances[app]
            app_params["mean"] = float(app_params["mean"])
            app_params["std"] = float(app_params["std"])
            app_params["mains_mean"] = float(app_params["mains_mean"])
            app_params["mains_std"] = float(app_params["mains_std"])
            params_to_save = {}
            params_to_save['appliance_params'] = app_params

            f = open(join(folder_name, app.replace(" ", "_") + ".json"), "w")
            f.write(json.dumps(params_to_save))

    def load_model(self, folder_name):
        #Get all the models trained in the given folder and load them.
        app_models = [f for f in listdir(folder_name) if isfile(join(folder_name, f)) and ".h5" in f ]
        for app in app_models:
            app_name = app.split(".")[0].replace("_", " ")
            self.model[app_name] = load_model(join(folder_name, app))

            f = open(join(folder_name, app_name.replace(" ", "_") + ".json"), "r")

            model_string = f.read().strip()
            params_to_load = json.loads(model_string)
            self.appliances[app_name] = params_to_load['appliance_params']

    def create_model(self, n_nodes, input_shape):
        model = Sequential()
        model.add(InputLayer(input_shape))
        model.add(GRU(160, return_sequences=True))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.2))
        model.add(GRU(64, return_sequences=True))
        model.add(Dense(96, activation='relu'))
        model.add(Dropout(0.2))
        model.add(GRU(96))
        model.add(Dense(224, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(1))

        model.compile(loss='mean_squared_error', metrics=["MeanAbsoluteError", "RootMeanSquaredError"], optimizer='adam')

        return model

    def create_transfer_model(self, transfer_path, input_shape, n_nodes=90):
        #Creates a specific model.
        model = Sequential()
        #Block 1
        model.add(GRU(n_nodes, input_shape=input_shape, return_sequences=True))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dropout(0.2))
        #Block 2
        model.add(GRU(n_nodes*2, return_sequences=True))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dropout(0.2))
        #Block 3
        model.add(GRU(int(n_nodes/2)))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dropout(0.2))
        #Dense Layer
        model.add(Dense(int(n_nodes/4), activation='relu'))
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dropout(0.2))
        #Classification Layer
        model.add(Dense(1))
        
        model.load_weights(transfer_path, skip_mismatch=True, by_name=True)

        for layer in model.layers:
            layer.trainable = False
            
        model.add(Dense(1))

        model.compile(loss='mean_squared_error', metrics=["MeanAbsoluteError", "RootMeanSquaredError"], optimizer='adam')

        return model

   