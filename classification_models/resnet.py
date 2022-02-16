import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=RuntimeWarning)

from os.path import join, isfile
from os import listdir

from tensorflow.keras.models import load_model
import tensorflow.keras as keras
from tensorflow.keras.callbacks import ModelCheckpoint


from sklearn.metrics import matthews_corrcoef, confusion_matrix, f1_score

import numpy as np
import json

#import sys
#sys.path.insert(1, "../utils")
#sys.path.insert(1, "../feature_extractors")

from generate_timeseries import generate_main_timeseries, generate_appliance_timeseries
from metrics import matthews_correlation

import utils
import plots

class ResNet():
    def __init__(self, params):
        #Variable that will store the models trained for each appliance.
        self.model = {}
        #Name used to identify the Model Name.
        self.MODEL_NAME = params.get('model_name', 'ResNet')
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
            "timestep": 2,
            "overlap": 178,
            'epochs' : 300,
            'batch_size' : 512,
            "on_treshold" : 50,
            'n_nodes' : 120
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
        
        for app_name, data in train_data.items():
            if( self.verbose != 0):
                print("Preparing Dataset for %s" % app_name)

            appliance_model = self.appliances[app_name]

            timewindow = appliance_model.get("timewindow", self.default_appliance['timewindow'])
            timestep = appliance_model["timestep"]
            overlap = appliance_model.get("overlap", self.default_appliance['overlap'])
            on_treshold = appliance_model.get("on_treshold", self.default_appliance['on_treshold'])
            n_nodes = appliance_model.get("n_nodes", self.default_appliance['n_nodes'])
            epochs = appliance_model.get("epochs", self.default_appliance['epochs'])
            batch_size = appliance_model.get("batch_size", self.default_appliance['batch_size'])
            mains_std = appliance_model.get("mains_std", None)
            mains_mean = appliance_model.get("mains_mean", None)

            if mains_mean is None:
                X_train, mains_mean, mains_std = generate_main_timeseries(data["mains"], timewindow, timestep, overlap)
                appliance_model["mains_mean"] = mains_mean
                appliance_model["mains_std"] = mains_std
            else:
                X_train = generate_main_timeseries(data["mains"], timewindow, timestep, overlap, mains_mean, mains_std)[0]  
            
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1, X_train.shape[2]))

            y_train = generate_appliance_timeseries(data["appliance"], True, timewindow, timestep, overlap, on_treshold=on_treshold)

            if cv_data is not None:
                X_cv = generate_main_timeseries(cv_data[app_name]["mains"], timewindow, timestep, overlap, mains_mean, mains_std)[0]
                X_cv= X_cv.reshape((X_cv.shape[0], X_cv.shape[1], 1, X_cv.shape[2]))
                y_cv = generate_appliance_timeseries(cv_data[app_name]["appliance"], True, timewindow, timestep, overlap, on_treshold=on_treshold)
                n_activations_cv = sum([ np.where(p == max(p))[0][0]  for p in y_cv ])

            n_activations_train = sum([ np.where(p == max(p))[0][0]  for p in y_train ])
            if( self.verbose == 2):
                print("-"*5 + "Train Info" + "-"*5)
                print("Nº of examples: ", str(X_train.shape[0]))
                print("Nº of activations: ", str(n_activations_train))
                print("On Percentage: ", str(n_activations_train/len(y_train) ))
                print("Off Percentage: ", str( (len(y_train) - n_activations_train)/len(y_train) ))
                if cv_data is not None:
                    print("-"*5 + "Cross Validation Info" + "-"*5)
                    print("Nº of examples: ", str(X_cv.shape[0]))
                    print("Nº of activations: ", str(n_activations_cv))
                    print("On Percentage: ", str(n_activations_cv/len(y_cv) ))
                    print("Off Percentage: ",  str( (len(y_cv) - n_activations_cv)/len(y_cv) ))

                print("Mains Mean: ", str(mains_mean))
                print("Mains Std: ", str(mains_std))
            
            if app_name in self.model:
                if self.verbose > 0:
                    print("Starting from previous step")
                model = self.model[app_name]
            else:
                model = self.create_model(n_nodes, (X_train.shape[1], X_train.shape[2], X_train.shape[3]))

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
                plots.plot_model_history_classification(json.loads(history), self.plots_folder + "/" + app_name + "/")

            model.load_weights(self.checkpoint_folder + "model_checkpoint_" + app_name.replace(" ", "_") + ".h5")
            
            self.model[app_name] = model

            if cv_data is not None:
                X = np.concatenate((X_train, X_cv), axis=0)
                y = np.concatenate((y_train, y_cv), axis=0)
            else:
                X = X_train
                y = y_train

            pred = self.model[app_name].predict(X)
            pred = [ np.where(p == max(p))[0][0]  for p in pred]         
            y = [ np.where(p == max(p))[0][0]  for p in y]

            tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
            mcc = matthews_corrcoef(y, pred)
            f1 = f1_score(y, pred)

            if self.verbose == 2:
                print("Training scores")    
                print("MCC: ", mcc )
                print("F1-Score: ", f1 )
                print("True Positives: ", tp)
                print("True Negatives: ", tn)  
                print("False Negatives: ", fn)  
                print("False Positives: ", fp)  
            
            if self.results_folder is not None:
                f = open(self.results_folder + "results_" + app_name.replace(" ", "_") + ".txt", "w")
                f.write("-"*5 + "Train Info" + "-"*5+ "\n")
                f.write("Nº of examples: "+ str(X.shape[0])+ "\n")
                f.write("Nº of activations: "+ str(n_activations_train)+ "\n")
                f.write("On Percentage: "+ str(n_activations_train/len(y_train))+ "\n")
                f.write("Off Percentage: "+ str((len(y_train) - n_activations_train)/len(y_train))+ "\n")
                f.write("-"*10+ "\n")
                if cv_data is not None:
                    f.write("-"*5 + "Cross Validation Info" + "-"*5+ "\n")
                    f.write("Nº of examples: "+ str(X_cv.shape[0])+ "\n")
                    f.write("Nº of activations: "+ str(n_activations_cv)+ "\n")
                    f.write("On Percentage: "+ str(n_activations_cv/len(y_cv))+ "\n")
                    f.write("Off Percentage: "+ str((len(y_cv) - n_activations_cv)/len(y_cv))+ "\n")
                f.write("-"*10+ "\n")
                if self.random_search:
                    #best_hp = tuner.get_best_hyperparameters()[0]
                    f.write("-"*5 + "Best Hyperparameters" + "-"*5+ "\n")
                    #f.write(str(best_hp.values)+"\n")
                    f.write("-"*10+ "\n")
                f.write("Mains Mean: " + str(mains_mean) + "\n")
                f.write("Mains Std: " + str(mains_std) + "\n")
                f.write("Train MCC: "+str(mcc)+ "\n")
                f.write("Train F1-Score: "+str(f1)+ "\n")
                f.write("True Positives: "+str(tp)+ "\n")
                f.write("True Negatives: "+str(tn)+ "\n")
                f.write("False Positives: "+str(fp)+ "\n")
                f.write("False Negatives: "+str(fn)+ "\n")
                f.close()
            
    def disaggregate_chunk(self, test_mains, test_appliances):
        
        appliance_powers_dict = {}
        
        for app_name, appliance_power in test_appliances:
            if self.verbose != 0:
                print("Preparing the Test Data for %s" % app_name)
            
            appliance_model = self.appliances.get(app_name, {})
        
            timewindow = appliance_model.get("timewindow", self.default_appliance['timewindow'])
            timestep = appliance_model["timestep"]
            overlap = appliance_model.get("overlap", self.default_appliance['overlap'])
            on_treshold = appliance_model.get("on_treshold", self.default_appliance['on_treshold'])
            mains_std = appliance_model.get("mains_std", None)
            mains_mean = appliance_model.get("mains_mean", None)

            X_test = generate_main_timeseries(test_mains, timewindow, timestep, overlap, mains_mean, mains_std)[0]
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1, X_test.shape[2]))

            if( self.verbose == 2):
                print("Nº of test examples", X_test.shape[0])

            if self.verbose != 0:
                print("Estimating power demand for '{}' in '{}'\n".format(app_name, self.MODEL_NAME))

            y_test = generate_appliance_timeseries(appliance_power, True, timewindow, timestep, overlap, on_treshold=on_treshold)

            n_activations = sum([ np.where(p == max(p))[0][0]  for p in y_test ])

            if( self.verbose != 0):
                print("Nº of examples: ", str(X_test.shape[0]))
                print("Nº of activations: ", str(n_activations))
                print("On Percentage: ", str(n_activations/len(y_test) ))
                print("Off Percentage: ", str( (len(y_test) - n_activations)/len(y_test) ))

            if self.verbose != 0:
                print("Estimating power demand for '{}' in '{}'\n".format(app_name, self.MODEL_NAME))
            
            pred = self.model[app_name].predict(X_test)
            pred = [ np.where(p == max(p))[0][0]  for p in pred]         
            y_test = [ np.where(p == max(p))[0][0]  for p in y_test]

            tn, fn, fp, tp = confusion_matrix(y_test, pred).ravel()
            mcc = matthews_corrcoef(y_test, pred)
            f1 = f1_score(y_test, pred)

            if self.verbose == 2:
                print("True Positives: ", tp)
                print("True Negatives: ", tn)  
                print("False Negatives: ", fn)  
                print("False Positives: ", fp)        
                print( "MCC: ", mcc)
                print( "F1-Score: ", f1)

            if self.results_folder is not None:
                f = open(self.results_folder + "results_" + app_name.replace(" ", "_") + ".txt", "a")
                f.write("-"*5 + "Test Info" + "-"*5+ "\n")
                f.write("Nº of examples: "+ str(X_test.shape[0])+ "\n")
                f.write("Nº of activations: "+ str(n_activations)+ "\n")
                f.write("On Percentage: "+ str(n_activations/len(y_test))+ "\n")
                f.write("Off Percentage: "+ str((len(y_test) - n_activations)/len(y_test))+ "\n")
                f.write("-"*10+ "\n")
                f.write("MCC: "+str(mcc) + "\n")
                f.write("F1-Score: "+str(f1) + "\n")
                f.write("True Positives: "+str(tp)+ "\n")
                f.write("True Negatives: "+str(tn)+ "\n")
                f.write("False Positives: "+str(fp)+ "\n")
                f.write("False Negatives: "+str(fn)+ "\n")
                f.close()

            appliance_powers_dict[app_name] = mcc

        return appliance_powers_dict

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

    def create_model(self, n_feature_maps, input_shape):

        input_layer = keras.layers.Input(input_shape)

        # BLOCK 1

        conv_x = keras.layers.Conv2D(filters=n_feature_maps, kernel_size=8, padding='same')(input_layer)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv2D(filters=n_feature_maps, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv2D(filters=n_feature_maps, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv2D(filters=n_feature_maps, kernel_size=1, padding='same')(input_layer)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_1 = keras.layers.add([shortcut_y, conv_z])
        output_block_1 = keras.layers.Activation('relu')(output_block_1)

        output_block_1 = keras.layers.Dropout(0.5)(output_block_1)

        # BLOCK 2

        conv_x = keras.layers.Conv2D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_1)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv2D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv2D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # expand channels for the sum
        shortcut_y = keras.layers.Conv2D(filters=n_feature_maps * 2, kernel_size=1, padding='same')(output_block_1)
        shortcut_y = keras.layers.BatchNormalization()(shortcut_y)

        output_block_2 = keras.layers.add([shortcut_y, conv_z])
        output_block_2 = keras.layers.Activation('relu')(output_block_2)

        output_block_2 = keras.layers.Dropout(0.5)(output_block_2)

        # BLOCK 3

        conv_x = keras.layers.Conv2D(filters=n_feature_maps * 2, kernel_size=8, padding='same')(output_block_2)
        conv_x = keras.layers.BatchNormalization()(conv_x)
        conv_x = keras.layers.Activation('relu')(conv_x)

        conv_y = keras.layers.Conv2D(filters=n_feature_maps * 2, kernel_size=5, padding='same')(conv_x)
        conv_y = keras.layers.BatchNormalization()(conv_y)
        conv_y = keras.layers.Activation('relu')(conv_y)

        conv_z = keras.layers.Conv2D(filters=n_feature_maps * 2, kernel_size=3, padding='same')(conv_y)
        conv_z = keras.layers.BatchNormalization()(conv_z)

        # no need to expand channels because they are equal
        shortcut_y = keras.layers.BatchNormalization()(output_block_2)

        output_block_3 = keras.layers.add([shortcut_y, conv_z])
        output_block_3 = keras.layers.Activation('relu')(output_block_3)

        output_block_3 = keras.layers.Dropout(0.5)(output_block_3)

        # FINAL

        gap_layer = keras.layers.GlobalAveragePooling2D()(output_block_3)

        output_layer = keras.layers.Dense(2, activation='softmax')(gap_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                        metrics=["accuracy", matthews_correlation])

        return model