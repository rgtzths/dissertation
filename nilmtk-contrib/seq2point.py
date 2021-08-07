from collections import OrderedDict
import numpy as np
import pandas as pd
from nilmtk.disaggregate import Disaggregator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Flatten
from tensorflow.keras.models import Sequential
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import os
import json

#import sys
#sys.path.insert(1, "../utils")
#sys.path.insert(1, "../feature_extractors")

import utils
import plots

class SequenceLengthError(Exception):
    pass

class ApplianceNotFoundError(Exception):
    pass

class Seq2Point(Disaggregator):

    def __init__(self, params):
        """
        Parameters to be specified for the model
        """

        self.MODEL_NAME = "Seq2Point"
        self.models = OrderedDict()
        self.file_prefix = params.get('file_prefix', "")
        self.verbose =  params.get('verbose', 0)
        self.chunk_wise_training = params.get('chunk_wise_training',False)
        self.sequence_length = params.get('sequence_length',99)
        self.n_epochs = params.get('n_epochs', 10 )
        self.batch_size = params.get('batch_size',512)
        self.appliance_params = params.get('appliance_params',{})
        self.mains_mean = params.get('mains_mean',None)
        self.mains_std = params.get('mains_std',None)
        self.appliances = params.get('appliances', {})
        self.load_model_path = params.get('pretrained-model-path',None)

        if self.sequence_length%2==0:
            print ("Sequence length should be odd!")
            raise (SequenceLengthError)

        self.training_history_folder = params.get("training_history_folder", None)
        self.plots_folder = params.get("plots_folder", None)
        self.results_folder = params.get("results_folder", None)

        if self.training_history_folder is not None:
            utils.create_path(self.training_history_folder)

        if self.plots_folder is not None:
            utils.create_path(self.plots_folder)

        if self.results_folder is not None:
            utils.create_path(self.results_folder)

        if self.load_model_path:
            self.load_model()

    def partial_fit(self, train_data, cv_data=None):
        if self.verbose > 0:
            print("...............Seq2Point partial_fit running...............")

        # If no appliance wise parameters are provided, then copmute them using the first chunk
        if len(self.appliance_params) == 0:
            self.set_appliance_params(train_data)

        if self.mains_mean is None:
            self.set_mains_params(list(train_data.values())[0]["mains"])

        for appliance_name, data in train_data.items():
            if( self.verbose != 0):
                print("Preparing Dataset for %s" % appliance_name)

            train_main, train_appliance = self.call_preprocessing(data["mains"], data["appliance"], appliance_name, 'train')

            train_main = pd.concat(train_main, axis=0).values.reshape((-1, self.sequence_length, 1))
            train_appliance = pd.concat(train_appliance, axis=0).values.reshape((-1, 1))
            
            if cv_data is not None:
                cv_main, cv_appliance = self.call_preprocessing(cv_data[appliance_name]["mains"], cv_data[appliance_name]["appliance"], appliance_name, 'train')
                
                cv_main = pd.concat(cv_main, axis=0).values.reshape((-1,self.sequence_length,1))
                cv_appliance = pd.concat(cv_appliance, axis=0).values.reshape((-1, 1))

            appliance_model = self.appliances.get(appliance_name, {})
            transfer_path = appliance_model.get("transfer_path", None)
            on_treshold = appliance_model.get("on_treshold", 50)
            mean = self.appliance_params[appliance_name]["mean"]
            std = self.appliance_params[appliance_name]["std"]

            binary_y = np.array([ 1 if x[0] > on_treshold else 0 for x in train_appliance*std + mean])
            
            train_positives = np.where(binary_y == 1)[0]

            train_n_activatons = train_positives.shape[0]
            train_on_examples = train_n_activatons / train_appliance.shape[0]
            train_off_examples = (train_appliance.shape[0] - train_n_activatons) / train_appliance.shape[0]

            if cv_data is not None:
                binary_y = np.array([ 1 if x[0] > on_treshold else 0 for x in cv_appliance*std + mean])
                    
                cv_positives = np.where(binary_y == 1)[0]

                cv_n_activatons = cv_positives.shape[0]
                cv_on_examples = cv_n_activatons / cv_appliance.shape[0]
                cv_off_examples = (cv_appliance.shape[0] - cv_n_activatons) / cv_appliance.shape[0]

            if( self.verbose == 2):
                print("-"*5 + "Train Info" + "-"*5)
                print("Nº of examples: ", str(train_appliance.shape[0]))
                print("Nº of activations: ", str(train_n_activatons))
                print("On Percentage: ", str(train_on_examples))
                print("Off Percentage: ", str(train_off_examples))
                if cv_data is not None:
                    print("-"*5 + "Cross Validation Info" + "-"*5)
                    print("Nº of examples: ", str(cv_appliance.shape[0]))
                    print("Nº of activations: ", str(cv_n_activatons))
                    print("On Percentage: ", str(cv_on_examples))
                    print("Off Percentage: ", str(cv_off_examples))
                print("-"*10)
                print("Mains Mean: ", str(self.mains_mean))
                print("Mains Std: ", str(self.mains_std))
                print(appliance_name + " Mean: ", str(mean))
                print(appliance_name + " Std: ", str(std))

            # Check if the appliance was already trained. If not then create a new model for it
            if appliance_name not in self.models:
                if transfer_path is not None:
                    if( self.verbose != 0):
                        print("Using transfer learning for ", appliance_name)
                    self.models[appliance_name] = self.create_transfer_model(transfer_path)
                else:
                    if( self.verbose != 0):
                        print("First model training for", appliance_name)
                    self.models[appliance_name] = self.return_network()
            # Retrain the particular appliance
            else:
                if( self.verbose != 0):
                    print("Started Retraining model for", appliance_name)

            model = self.models[appliance_name]

            filepath = self.file_prefix + "{}.h5".format("_".join(appliance_name.split()))

            verbose = 1 if self.verbose >= 1 else 0

            checkpoint = ModelCheckpoint(filepath,monitor='val_loss',verbose=verbose,save_best_only=True,mode='min')

            if cv_data is not None:
                history = model.fit(train_main, 
                        train_appliance,
                        epochs=self.n_epochs, 
                        batch_size=self.batch_size,
                        shuffle=False,
                        callbacks=[checkpoint],
                        validation_data=(cv_main, cv_appliance),
                        verbose=verbose
                        )        
            else:
                history = model.fit(train_main, 
                    train_appliance,
                    epochs=self.n_epochs, 
                    batch_size=self.batch_size,
                    shuffle=False,
                    callbacks=[checkpoint],
                    validation_split=0.15,
                    verbose=verbose
                    )

            model.load_weights(filepath)

            history = json.dumps(history.history)
            
            if self.training_history_folder is not None:
                f = open(self.training_history_folder + "history_"+appliance_name.replace(" ", "_")+".json", "w")
                f.write(history)
                f.close()

            if self.plots_folder is not None:
                utils.create_path(self.plots_folder + "/" + appliance_name + "/")
                plots.plot_model_history_regression(json.loads(history), self.plots_folder + "/" + appliance_name + "/")

            #Gets the trainning data score
            #Concatenates training and cross_validation
            X = np.concatenate((train_main, cv_main), axis=0)
            y = np.concatenate((train_appliance, cv_appliance), axis=0)

            pred = self.models[appliance_name].predict(X) * std + mean

            train_rmse = math.sqrt(mean_squared_error(y * std + mean, pred))
            train_mae = mean_absolute_error(y * std + mean, pred)

            if self.verbose == 2:
                print("Training scores")    
                print("RMSE: ", train_rmse )
                print("MAE: ", train_mae )
            
            if self.results_folder is not None:
                f = open(self.results_folder + "results_" + appliance_name.replace(" ", "_") + ".txt", "w")
                f.write("-"*5 + "Train Info" + "-"*5)
                f.write("Nº of examples: "+ str(train_appliance.shape[0]))
                f.write("Nº of activations: "+ str(train_n_activatons))
                f.write("On Percentage: "+ str(train_on_examples))
                f.write("Off Percentage: "+ str(train_off_examples))
                if cv_data is not None:
                    f.write("-"*5 + "Cross Validation Info" + "-"*5)
                    f.write("Nº of examples: "+ str(cv_appliance.shape[0]))
                    f.write("Nº of activations: "+ str(cv_n_activatons))
                    f.write("On Percentage: "+ str(cv_on_examples))
                    f.write("Off Percentage: "+ str(cv_off_examples))
                f.write("-"*10)
                f.write("Mains Mean: " + str(self.mains_mean) + "\n")
                f.write("Mains Std: " + str(self.mains_std) + "\n")
                f.write(appliance_name + " Mean: " + str(mean) + "\n")
                f.write(appliance_name + " Std: " + str(std) + "\n")
                f.write("Train RMSE: "+str(train_rmse)+ "\n")
                f.write("Train MAE: "+str(train_mae)+ "\n")
                f.close()

    def disaggregate_chunk(self, test_mains, app_name):

        test_predictions = []
        disggregation_dict = {}
        
        test_main_list = self.call_preprocessing(test_mains, app_df_list=None, appliance_name=None, method='test')
        test_main = pd.concat(test_main_list, axis=0).values.reshape((-1,self.sequence_length,1))

        app_mean = self.appliance_params[app_name]['mean']
        app_std = self.appliance_params[app_name]['std']
        
        prediction = self.models[app_name].predict(test_main,batch_size=self.batch_size).flatten()* app_std + app_mean 
        prediction = np.where(prediction>0, prediction,0)

        disggregation_dict[app_name] = pd.Series(prediction)
        results = pd.DataFrame(disggregation_dict,dtype='float32')

        test_predictions.append(results)
        return test_predictions

    def save_model(self, save_model_path):
        for appliance_name in self.models:
            print ("Saving model for ", appliance_name)
            self.models[appliance_name].save_weights(os.path.join(save_model_path,appliance_name+".h5"))

    def load_model(self):
        print ("Loading the model using the pretrained-weights")        
        model_folder = self.load_model_path

        for appliance_name in self.appliance_params:
            self.models[appliance_name] = self.return_network()
            self.models[appliance_name].load_weights(os.path.join(model_folder,appliance_name+".h5"))

    def return_network(self):
        # Model architecture
        model = Sequential()
        model.add(Conv1D(30,10,activation="relu",input_shape=(self.sequence_length,1),strides=1))
        model.add(Conv1D(30, 8, activation='relu', strides=1))
        model.add(Conv1D(40, 6, activation='relu', strides=1))
        model.add(Conv1D(50, 5, activation='relu', strides=1))
        model.add(Dropout(.2))
        model.add(Conv1D(50, 5, activation='relu', strides=1))
        model.add(Dropout(.2))
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(.2))
        model.add(Dense(1))
        
        model.compile(loss='mean_squared_error', metrics=["MeanAbsoluteError", "RootMeanSquaredError"], optimizer='adam')
        
        return model

    def create_transfer_model(self, transfer_path):
        # Model architecture
        model = Sequential()
        model.add(Conv1D(30,10,activation="relu",input_shape=(self.sequence_length,1),strides=1))
        model.add(Conv1D(30, 8, activation='relu', strides=1))
        model.add(Conv1D(40, 6, activation='relu', strides=1))
        model.add(Conv1D(50, 5, activation='relu', strides=1))
        model.add(Dropout(.2))
        model.add(Conv1D(50, 5, activation='relu', strides=1))
        model.add(Dropout(.2))
        model.add(Flatten())

        model.load_weights(transfer_path, skip_mismatch=True, by_name=True)

        for layer in model.layers:
            layer.trainable = False
        
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(.2))
        model.add(Dense(1))
        
        model.compile(loss='mean_squared_error', metrics=["MeanAbsoluteError", "RootMeanSquaredError"], optimizer='adam')

        return model

    def call_preprocessing(self, mains_lst, app_df_list, appliance_name, method):

        if method == 'train':
            # Preprocessing for the train data
            mains_df_list = []
            for mains in mains_lst:
                new_mains = mains.values.flatten()
                n = self.sequence_length
                units_to_pad = n // 2
                new_mains = np.pad(new_mains,(units_to_pad,units_to_pad),'constant',constant_values=(0,0))
                new_mains = np.array([new_mains[i:i + n] for i in range(len(new_mains) - n + 1)])
                new_mains = (new_mains - self.mains_mean) / self.mains_std
                mains_df_list.append(pd.DataFrame(new_mains))
            
            processed_appliance_dfs = []
            app_mean = self.appliance_params[appliance_name]['mean']
            app_std = self.appliance_params[appliance_name]['std']
            for app_df in app_df_list:
                new_app_readings = app_df.values.reshape((-1, 1))
                new_app_readings = (new_app_readings - app_mean) / app_std  
                processed_appliance_dfs.append(pd.DataFrame(new_app_readings))

            return mains_df_list, processed_appliance_dfs

        else:
            # Preprocessing for the test data
            mains_df_list = []

            for mains in mains_lst:
                new_mains = mains.values.flatten()
                n = self.sequence_length
                units_to_pad = n // 2
                new_mains = np.pad(new_mains,(units_to_pad,units_to_pad),'constant',constant_values=(0,0))
                new_mains = np.array([new_mains[i:i + n] for i in range(len(new_mains) - n + 1)])
                new_mains = (new_mains - self.mains_mean) / self.mains_std
                mains_df_list.append(pd.DataFrame(new_mains))
            return mains_df_list

    def set_appliance_params(self, train_data):
        # Find the parameters using the first
        for app_name, data in train_data.items():
            l = np.array(pd.concat(data["appliance"],axis=0))
            app_mean = np.mean(l)
            app_std = np.std(l)
            if app_std<1:
                app_std = 100
            self.appliance_params.update({app_name:{'mean':app_mean,'std':app_std}})
        print (self.appliance_params)

    def set_mains_params(self, train_mains):
        l = np.array(pd.concat(train_mains, axis=0))
        self.mains_mean = np.mean(l, axis=0)
        self.mains_std = np.std(l, axis=0)
