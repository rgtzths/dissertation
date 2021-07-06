import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=RuntimeWarning)

from os.path import join, isfile
from os import listdir
import json

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, GRU, Dropout, Input, Attention, Flatten
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from scipy.stats import randint
from sklearn.metrics import matthews_corrcoef, confusion_matrix, make_scorer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import shuffle

import numpy as np

#import sys
#sys.path.insert(1, "../utils")
#sys.path.insert(1, "../feature_extractors")

from generate_timeseries import generate_main_timeseries, generate_appliance_timeseries
from matthews_correlation import matthews_correlation

import utils
import plots

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

class GRU_RNN():
    def __init__(self, params):
        #Variable that will store the models trained for each appliance.
        self.model = {}
        #Name used to identify the Model Name.
        self.MODEL_NAME = params.get('model_name', 'GRU')
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

                y_train = generate_appliance_timeseries(appliance_power, True, timewindow, timestep, overlap)

                X_train, y_train = shuffle(X_train, y_train, random_state=0)

                X_train, X_cv = X_train[:-int(X_train.shape[0]*self.cv)], X_train[-int(X_train.shape[0]*self.cv) :]

                y_train, y_cv = y_train[:-int(y_train.shape[0]*self.cv)], y_train[-int(y_train.shape[0]*self.cv) :]

                train_cutoff = len(X_train) % batch_size
                cv_cutoff = len(X_cv) % batch_size

                X_train, y_train = X_train[:-train_cutoff], y_train[:-train_cutoff]

                X_cv, y_cv = X_cv[:-cv_cutoff], y_cv[:-cv_cutoff]

                if( self.verbose != 0):
                    print("Nº of positive examples ", sum([ np.where(p == max(p))[0][0]  for p in y_train]))
                    print("Nº of negative examples ", y_train.shape[0]-sum([ np.where(p == max(p))[0][0]  for p in y_train ]))

                if self.verbose != 0:
                    print("Training ", app_name, " in ", self.MODEL_NAME, " model\n", end="\r")
                
                #Checks if the model already exists and if it doesn't creates a new one.
                if app_name in self.model:
                    model = self.model[app_name]
                else:
                    model = self.create_model(n_nodes, (X_train.shape[1], X_train.shape[2]), batch_size)

                checkpoint = ModelCheckpoint(
                                self.checkpoint_folder + "model_checkpoint_" + app_name.replace(" ", "_") + ".h5", 
                                monitor='val_loss', 
                                verbose=self.verbose, 
                                save_best_only=True, 
                                mode='min'
                                )
                
                #Fits the model to the training data.
                history = model.fit(X_train, 
                        y_train, 
                        epochs=epochs, 
                        batch_size=batch_size,
                        verbose=self.verbose,
                        callbacks=[checkpoint],
                        validation_data=(X_cv, y_cv)
                        )

                history = json.dumps(history.history)

                if self.training_history_folder is not None:
                    f = open(self.training_history_folder + "history_"+app_name.replace(" ", "_")+".json", "w")
                    f.write(history)
                    f.close()

                if self.plots_folder is not None:
                    utils.create_path(self.plots_folder + "/" + app_name + "/")
                    plots.plot_model_history(json.loads(history), self.plots_folder + "/" + app_name + "/")

                model.load_weights(self.checkpoint_folder + "model_checkpoint_" + app_name.replace(" ", "_") + ".h5")
                
                #Stores the trained model.
                self.model[app_name] = model

                #Gets the trainning data score
                pred = self.model[app_name].predict(X_train)
                pred = [ np.where(p == max(p))[0][0]  for p in pred ]
                
                y_train = [ np.where(p == max(p))[0][0]  for p in y_train ]
                
                tn, fp, fn, tp = confusion_matrix(y_train, pred).ravel()
                mcc = matthews_corrcoef(y_train, pred)

                if self.verbose == 2:
                    print("Training data scores")
                    print("True Positives: ", tp)
                    print("True Negatives: ", tn)  
                    print("False Negatives: ", fn)  
                    print("False Positives: ", fp)        
                    print("MCC: ", mcc )
                
                if self.results_folder is not None:
                    f = open(self.results_folder + "results_" + app_name.replace(" ", "_") + ".txt", "w")
                    f.write("Nº of positive examples for training: " + str(sum(y_train)) + "\n")
                    f.write("Nº of negative examples for training: " + str(len(y_train)- sum(y_train)) + "\n")
                    f.write("Data Mean: " + str(self.mains_mean) + "\n")
                    f.write("Data Std: " + str(self.mains_std) + "\n")
                    f.write("Train MCC: "+str(mcc)+ "\n")
                    f.write("True Positives: "+str(tp)+ "\n")
                    f.write("True Negatives: "+str(tn)+ "\n")
                    f.write("False Positives: "+str(fp)+ "\n")
                    f.write("False Negatives: "+str(fn)+ "\n")
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
                print("Nº of positive examples ", sum([ np.where(p == max(p))[0][0]  for p in y_test ]))
                print("Nº of negative examples ", y_test.shape[0]-sum([ np.where(p == max(p))[0][0]  for p in y_test ]))

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

            if self.results_folder is not None:
                f = open(self.results_folder + "results_" + app_name.replace(" ", "_") + ".txt", "a")
                f.write("Nº of positive examples for test: " + str(sum(y_test)) + "\n")
                f.write("Nº of negative examples for test: " + str(len(y_test)- sum(y_test)) + "\n")
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

    def create_model(self, n_nodes, input_shape, batch_size):
        #Creates a specific model.
        input_layer = Input(shape=input_shape, batch_size=batch_size)

        encoder_output, hidden_state = GRU(n_nodes, return_sequences=True, return_state=True)(input_layer)
        
        attention_input = [encoder_output, hidden_state]

        encoder_output = Attention()(attention_input)

        flatten_data = Flatten()(encoder_output)

        dense_layer = Dense(int(n_nodes/2), activation='relu')(flatten_data)
        
        dropout_layer = Dropout(0.1)(dense_layer)

        output_layer = Dense(2, activation='softmax')(dropout_layer)

        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(optimizer=Adam(0.00001), loss='categorical_crossentropy', metrics=["accuracy", matthews_correlation])

        return model
