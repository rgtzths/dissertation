from warnings import warn
from nilmtk.disaggregate import Disaggregator
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Reshape, Flatten
import pandas as pd
import numpy as np
from collections import OrderedDict 
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as  plt
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow.keras.backend as K
from statistics import mean
import os
import json


class DAE(Disaggregator):

    def __init__(self, params):
        """
        Iniititalize the moel with the given parameters
        """
        self.MODEL_NAME = "DAE"
        self.file_prefix = "{}-temp-weights".format(self.MODEL_NAME.lower())
        self.chunk_wise_training = params.get('chunk_wise_training',False)
        self.sequence_length = params.get('sequence_length',99)
        self.n_epochs = params.get('n_epochs', 10)
        self.batch_size = params.get('batch_size',512)
        self.mains_mean = params.get('mains_mean',1000)
        self.mains_std = params.get('mains_std',600)
        self.appliance_params = params.get('appliance_params',{})
        self.save_model_path = params.get('save-model-path', None)
        self.load_model_path = params.get('pretrained-model-path',None)
        self.models = OrderedDict()
        if self.load_model_path:
            self.load_model()


    def partial_fit(self, train_data, cv_data=None):
        """
        The partial fit function
        """

        # If no appliance wise parameters are specified, then they are computed from the data
        if len(self.appliance_params) == 0:
            self.set_appliance_params(train_data)

        # To preprocess the data and bring it to a valid shape
        for appliance_name, data in train_data.items():
            train_main, train_appliance = self.call_preprocessing(data["mains"], data["appliance"], appliance_name, 'train')

            train_main = pd.concat(train_main, axis=0).values.reshape((-1, self.sequence_length, 1))
            train_appliance = pd.concat(train_appliance, axis=0).values.reshape((-1, self.sequence_length, 1))

            if cv_data is not None:
                cv_main, cv_appliance = self.call_preprocessing(cv_data[appliance_name]["mains"], cv_data[appliance_name]["appliance"], appliance_name, 'train')

                cv_main = pd.concat(cv_main, axis=0).values.reshape((-1, self.sequence_length, 1))                
                cv_appliance = pd.concat(cv_appliance, axis=0).values.reshape((-1, self.sequence_length, 1))

            if appliance_name not in self.models:
                print("First model training for", appliance_name)
                self.models[appliance_name] = self.return_network()
                print(self.models[appliance_name].summary())

            print("Started Retraining model for", appliance_name)
            model = self.models[appliance_name]
            filepath = self.file_prefix + "-{}-epoch{}.h5".format(
                    "_".join(appliance_name.split()),
                    0,
            )
            checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
            if cv_data is not None:
                model.fit(
                    train_main, train_appliance,
                    validation_data=(cv_main, cv_appliance),
                    batch_size=self.batch_size,
                    epochs=self.n_epochs,
                    callbacks=[ checkpoint ],
                    shuffle=False,
                )
            else:
                model.fit(
                    train_main, train_appliance,
                    validation_split=.15,
                    batch_size=self.batch_size,
                    epochs=self.n_epochs,
                    callbacks=[ checkpoint ],
                    shuffle=False,
                )
            model.load_weights(filepath)

        if self.save_model_path:
            self.save_model()

    def load_model(self):
        print ("Loading the model using the pretrained-weights")        
        model_folder = self.load_model_path
        with open(os.path.join(model_folder, "model.json"), "r") as f:
            model_string = f.read().strip()
            params_to_load = json.loads(model_string)


        self.sequence_length = int(params_to_load['sequence_length'])
        self.mains_mean = params_to_load['mains_mean']
        self.mains_std = params_to_load['mains_std']
        self.appliance_params = params_to_load['appliance_params']

        for appliance_name in self.appliance_params:
            self.models[appliance_name] = self.return_network()
            self.models[appliance_name].load_weights(os.path.join(model_folder,appliance_name+".h5"))


    def save_model(self, path):
        
        params_to_save = {}
        params_to_save['appliance_params'] = self.appliance_params
        params_to_save['sequence_length'] = self.sequence_length
        params_to_save['mains_mean'] = self.mains_mean
        params_to_save['mains_std'] = self.mains_std
        for appliance_name in self.models:
            print ("Saving model for ", appliance_name)
            self.models[appliance_name].save_weights(os.path.join(path,appliance_name+".h5"))

        with open(os.path.join(path,'model.json'),'w') as file:
            file.write(json.dumps(params_to_save))



    def disaggregate_chunk(self, test_mains, appliance):

        test_main_list = self.call_preprocessing(test_mains, app_df_list=None, appliance_name=appliance, method='test')

        test_predictions = []
        for test_main in test_main_list:
            test_main = test_main.values
            test_main = test_main.reshape((-1,self.sequence_length,1))
            disggregation_dict = {}
            
            prediction = self.models[appliance].predict(test_main,batch_size=self.batch_size)
            app_mean = self.appliance_params[appliance]['mean']
            app_std = self.appliance_params[appliance]['std']
            prediction = self.denormalize_output(prediction,app_mean,app_std)
            valid_predictions = prediction.flatten()
            valid_predictions = np.where(valid_predictions>0,valid_predictions,0)
            series = pd.Series(valid_predictions)
            disggregation_dict[appliance] = series
            results = pd.DataFrame(disggregation_dict,dtype='float32')
            test_predictions.append(results)
        return test_predictions
            
    def return_network(self):
        model = Sequential()
        model.add(Conv1D(8, 4, activation="linear", input_shape=(self.sequence_length, 1), padding="same", strides=1))
        model.add(Flatten())
        model.add(Dense((self.sequence_length)*8, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense((self.sequence_length)*8, activation='relu'))
        model.add(Reshape(((self.sequence_length), 8)))
        model.add(Conv1D(1, 4, activation="linear", padding="same", strides=1))
        model.compile(loss='mse', optimizer='adam')
        return model

    def call_preprocessing(self, mains_lst, app_df_list, appliance_name, method):
        sequence_length  = self.sequence_length
        app_mean = self.appliance_params[appliance_name]['mean']
        app_std = self.appliance_params[appliance_name]['std']

        if method=='train':
            processed_mains = []
            for mains in mains_lst:                
                mains = self.normalize_input(mains.values,sequence_length,self.mains_mean,self.mains_std,True)
                processed_mains.append(pd.DataFrame(mains))

            
            
            processed_app_dfs = []
            for app_df in app_df_list:
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
        return (windowed_x/std).reshape((-1,sequence_length))

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
            self.appliance_params.update({app_name:{'mean':float(app_mean),'std':float(app_std)}})

