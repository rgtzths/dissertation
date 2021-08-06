from nilmtk.disaggregate import Disaggregator
from tensorflow.keras.layers import Conv1D, Dense, Reshape, Flatten
import pandas as pd
import numpy as np
from collections import OrderedDict 
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint
import os
import json

#import sys
#sys.path.insert(1, "../utils")
#sys.path.insert(1, "../feature_extractors")

import utils
import plots


class DAE(Disaggregator):

    def __init__(self, params):
        """
        Iniititalize the moel with the given parameters
        """
        self.MODEL_NAME = "DAE"
        self.file_prefix = params.get('file_prefix', "")
        self.chunk_wise_training = params.get('chunk_wise_training',False)
        self.sequence_length = params.get('sequence_length',99)
        self.n_epochs = params.get('n_epochs', 10)
        self.batch_size = params.get('batch_size',512)
        self.mains_mean = params.get('mains_mean',None)
        self.mains_std = params.get('mains_std',None)
        self.appliance_params = params.get('appliance_params',{})
        self.load_model_path = params.get('pretrained-model-path',None)
        self.appliances = params.get('appliances', {})
        self.on_treshold = params.get('on_treshold', 50)

        self.training_history_folder = params.get("training_history_folder", None)
        if self.training_history_folder is not None:
            utils.create_path(self.training_history_folder)

        self.plots_folder = params.get("plots_folder", None)
        if self.plots_folder is not None:
            utils.create_path(self.plots_folder)

        self.models = OrderedDict()
        if self.load_model_path:
            self.load_model()


    def partial_fit(self, train_data, cv_data=None, do_preprocessing=True, current_epoch=0, **load_kwargs):
        """
        The partial fit function
        """
        print("...............DAE partial_fit running...............")
        
        # If no appliance wise parameters are specified, then they are computed from the data
        if len(self.appliance_params) == 0:
            self.set_appliance_params(train_data)

        if self.mains_mean is None:
            self.set_mains_params(list(train_data.values())[0]["mains"])

        for appliance_name, data in train_data.items():
            if do_preprocessing:
                print("Preprocessing")
                train_main, train_appliance = self.call_preprocessing(data["mains"], data["appliance"], appliance_name, 'train')
            
            train_main = pd.concat(train_main, axis=0).values.reshape((-1, self.sequence_length, 1))
            train_appliance = pd.concat(train_appliance, axis=0).values.reshape((-1, self.sequence_length, 1))

            if cv_data is not None:
                if do_preprocessing:
                    print("Preprocessing")
                    cv_main, cv_appliance = self.call_preprocessing(cv_data[appliance_name]["mains"], cv_data[appliance_name]["appliance"], appliance_name, 'train')
                
                cv_main = pd.concat(cv_main, axis=0).values.reshape((-1, self.sequence_length, 1))                
                cv_appliance = pd.concat(cv_appliance, axis=0).values.reshape((-1, self.sequence_length, 1))

            
            appliance_model = self.appliances.get(appliance_name, {})
            transfer_path = appliance_model.get("transfer_path", None)

            if appliance_name not in self.models:
                if transfer_path is not None:
                    print("Using transfer learning for ", appliance_name)
                    model = self.create_transfer_model(transfer_path)
                else:
                    print("First model training for", appliance_name)
                    model = self.return_network()
                
            else:
                print("Started Retraining model for", appliance_name)
                model = self.models[appliance_name]

            filepath = self.file_prefix + "{}.h5".format("_".join(appliance_name.split()))
            checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

            if cv_data is not None:
                history = model.fit(train_main, 
                        train_appliance,
                        epochs=self.n_epochs, 
                        batch_size=self.batch_size,
                        shuffle=False,
                        callbacks=[checkpoint],
                        validation_data=(cv_main, cv_appliance),
                        verbose=1
                        )        
            else:
                history = model.fit(train_main, 
                    train_appliance,
                    epochs=self.n_epochs, 
                    batch_size=self.batch_size,
                    shuffle=False,
                    callbacks=[checkpoint],
                    validation_split=0.15,
                    verbose=1
                    )

            model.load_weights(filepath)

            self.models[appliance_name] = model
            history = json.dumps(history.history)

            if self.training_history_folder is not None:
                f = open(self.training_history_folder + "history_"+appliance_name.replace(" ", "_")+".json", "w")
                f.write(history)
                f.close()

            if self.plots_folder is not None:
                utils.create_path(self.plots_folder + "/" + appliance_name + "/")
                plots.plot_model_history_regression(json.loads(history), self.plots_folder + "/" + appliance_name + "/")

    def load_model(self):
        print ("Loading the model using the pretrained-weights")        
        model_folder = self.load_model_path

        for appliance_name in self.appliance_params:
            self.models[appliance_name] = self.return_network()
            self.models[appliance_name].load_weights(os.path.join(model_folder,appliance_name+".h5"))


    def save_model(self, save_model_path):

        if not os.path.isdir(save_model_path):
            os.makedirs(save_model_path)    

        for appliance_name in self.models:
            print ("Saving model for ", appliance_name)
            self.models[appliance_name].save_weights(os.path.join(save_model_path,appliance_name+".h5"))



    def disaggregate_chunk(self, test_mains, app_name, do_preprocessing=True):
        if do_preprocessing:
            test_main_list = self.call_preprocessing(test_mains, app_df_list=None, appliance_name=None, method='test')

        test_predictions = []
        for test_main in test_main_list:
            test_main = test_main.values
            test_main = test_main.reshape((-1,self.sequence_length,1))
            disggregation_dict = {}
            
            prediction = self.models[app_name].predict(test_main,batch_size=self.batch_size)
            app_mean = self.appliance_params[app_name]['mean']
            app_std = self.appliance_params[app_name]['std']
            prediction = self.denormalize_output(prediction,app_mean,app_std)
            valid_predictions = prediction.flatten()
            valid_predictions = np.where(valid_predictions>0,valid_predictions,0)
            series = pd.Series(valid_predictions)
            disggregation_dict[app_name] = series
            results = pd.DataFrame(disggregation_dict,dtype='float32')
            test_predictions.append(results)
        return test_predictions
            
    def return_network(self):
        model = Sequential()
        model.add(Conv1D(8, 4, activation="linear", input_shape=(self.sequence_length, 1), padding="same", strides=1))
        model.add(Flatten())
        model.add(Dense(2400, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense((self.sequence_length)*8, activation='relu'))
        model.add(Reshape(((self.sequence_length), 8)))
        model.add(Conv1D(1, 4, activation="linear", padding="same", strides=1))
        
        model.compile(loss='mean_squared_error', metrics=["MeanAbsoluteError", "RootMeanSquaredError"], optimizer='adam')
        
        return model

    def create_transfer_model(self, transfer_path):
        model = Sequential()
        model.add(Conv1D(8, 4, activation="linear", input_shape=(self.sequence_length, 1), padding="same", strides=1))
        model.add(Flatten())
        model.add(Dense(2400, activation='relu'))
        model.add(Dense(128, activation='relu'))

        model.load_weights(transfer_path, skip_mismatch=True, by_name=True)

        for layer in model.layers:
            layer.trainable = False

        model.add(Dense((self.sequence_length)*8, activation='relu'))
        model.add(Reshape(((self.sequence_length), 8)))
        model.add(Conv1D(1, 4, activation="linear", padding="same", strides=1))
        
        model.compile(loss='mean_squared_error', metrics=["MeanAbsoluteError", "RootMeanSquaredError"], optimizer='adam')
        return model

    def call_preprocessing(self, mains_lst, app_df_list, appliance_name, method):
        sequence_length  = self.sequence_length
        if method=='train':
            processed_mains = []
            for mains in mains_lst:                
                mains = self.normalize_input(mains.values,sequence_length,self.mains_mean,self.mains_std,True)
                processed_mains.append(pd.DataFrame(mains))

            processed_app_dfs = []
            for app_df in app_df_list:
                app_mean = self.appliance_params[appliance_name]['mean']
                app_std = self.appliance_params[appliance_name]['std']
                data = self.normalize_output(app_df.values, sequence_length,app_mean,app_std,True)
                processed_app_dfs.append(pd.DataFrame(data))      

            return processed_mains, processed_app_dfs

        if method=='test':
            processed_mains = []
            for mains in mains_lst:                
                mains = self.normalize_input(mains.values,sequence_length,self.mains_mean,self.mains_std,False)
                processed_mains.append(pd.DataFrame(mains))
            return processed_mains
    
        
    def normalize_input(self,data,sequence_length, mean, std, overlapping=False):
        n = sequence_length
        excess_entries =  sequence_length - (data.size % sequence_length)       
        lst = np.array([0] * excess_entries)
        arr = np.concatenate((data.flatten(), lst),axis=0)   
        if overlapping:
            windowed_x = np.array([ arr[i:i+n] for i in range(len(arr)-n+1) ])
        else:
            windowed_x = arr.reshape((-1,sequence_length))
        windowed_x = windowed_x - mean
        windowed_x = windowed_x/std
        return (windowed_x).reshape((-1,sequence_length))

    def normalize_output(self,data,sequence_length, mean, std, overlapping=False):
        n = sequence_length
        excess_entries =  sequence_length - (data.size % sequence_length)       
        lst = np.array([0] * excess_entries)
        arr = np.concatenate((data.flatten(), lst),axis=0) 
        if overlapping:  
            windowed_y = np.array([ arr[i:i+n] for i in range(len(arr)-n+1) ])
        else:
            windowed_y = arr.reshape((-1,sequence_length))        
        windowed_y = windowed_y - mean
        return (windowed_y/std).reshape((-1,sequence_length))

    def denormalize_output(self,data,mean,std):
        return mean + data*std
    
    def set_appliance_params(self,train_data):
        for app_name, data in train_data.items():
            l = np.array(pd.concat(data["appliance"],axis=0))
            app_mean = np.mean(l)
            app_std = np.std(l)
            if app_std<1:
                app_std = 100
            self.appliance_params.update({app_name:{'mean':app_mean,'std':app_std}})

    def set_mains_params(self, train_mains):
        l = np.array(pd.concat(train_mains, axis=0))
        self.mains_mean = np.mean(l, axis=0)
        self.mains_std = np.std(l, axis=0)

