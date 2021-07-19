import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=RuntimeWarning)

from os.path import join, isfile
from os import listdir
import json
import math

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, GRU, LeakyReLU, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

import pandas as pd

#import sys
#sys.path.insert(1, "../utils")
#sys.path.insert(1, "../feature_extractors")

from generate_timeseries import generate_main_timeseries, generate_appliance_timeseries

import utils
import plots

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

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

        self.appliances = params.get('appliances', {})

        self.default_appliance = {
            "timewindow": 180,
            "overlap": 178,
            "timestep": 2,
            'epochs' : 1,
            'batch_size' : 1024,
            'n_nodes' : 90
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

                X_train, self.mains_mean, self.mains_std = generate_main_timeseries(train_mains, timewindow, timestep, overlap)  

                y_train = generate_appliance_timeseries(appliance_power, False, timewindow, timestep, overlap)

                X_train, X_cv, y_train, y_cv = train_test_split(X_train, y_train, test_size=self.cv, stratify=[ 1 if x > 80 else 0 for x in y_train])

                if( self.verbose != 0):
                     print("Nº of examples ", str(X_train.shape[0]))

                if self.verbose != 0:
                    print("Training ", app_name, " in ", self.MODEL_NAME, " model\n", end="\r")
                
                model = self.create_model(n_nodes, (X_train.shape[1], X_train.shape[2]))

                checkpoint = ModelCheckpoint(
                                self.checkpoint_folder + "model_checkpoint_" + app_name.replace(" ", "_") + ".h5", 
                                monitor='val_loss', 
                                verbose=self.verbose, 
                                save_best_only=True, 
                                mode='min'
                                )

                #Fits the model to the training data.
                history = model.fit( X_train, 
                        y_train, 
                        epochs=epochs, 
                        batch_size=batch_size, 
                        verbose=self.verbose, 
                        shuffle=True,
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
                pred = self.model[app_name].predict(X_train)

                rmse = math.sqrt(mean_squared_error(y_train, pred))
                mae = mean_absolute_error(y_train, pred)

                if self.verbose == 2:
                    print("Training data scores")    
                    print("RMSE: ", rmse )
                    print("MAE: ", mae )
                
                if self.results_folder is not None:
                    f = open(self.results_folder + "results_" + app_name.replace(" ", "_") + ".txt", "w")
                    f.write("Nº of examples for training: " + str(y_train.shape[0]) + "\n")
                    f.write("Data Mean: " + str(self.mains_mean) + "\n")
                    f.write("Data Std: " + str(self.mains_std) + "\n")
                    f.write("Train RMSE: "+str(rmse)+ "\n")
                    f.write("Train MAE: "+str(mae)+ "\n")
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

            X_test = generate_main_timeseries(test_mains, timewindow, timestep, overlap, self.mains_mean, self.mains_std)[0] 

            if( self.verbose != 0):
                print("Nº of examples", X_test.shape[0])

            if self.verbose != 0:
                print("Estimating power demand for '{}' in '{}'\n".format(app_name, self.MODEL_NAME))
            
            pred = self.model[app_name].predict(X_test)
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
        model.compile(optimizer=Adam(0.00001), loss='mean_squared_error', metrics=["MeanAbsoluteError", "RootMeanSquaredError"])

        return model
