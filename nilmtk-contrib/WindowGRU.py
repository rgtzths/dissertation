from collections import OrderedDict
import numpy as np
import pandas as pd
from nilmtk.disaggregate import Disaggregator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Conv1D, GRU, Bidirectional, Dropout
from tensorflow.keras.models import Sequential
import random
import json
import os

#import sys
#sys.path.insert(1, "../utils")
#sys.path.insert(1, "../feature_extractors")

import utils
import plots

class SequenceLengthError(Exception):
    pass

class ApplianceNotFoundError(Exception):
    pass

class WindowGRU(Disaggregator):

    def __init__(self, params):

        self.MODEL_NAME = "WindowGRU"
        self.models = OrderedDict()
        self.file_prefix = params.get('file_prefix', "")
        self.save_model_path = params.get('save-model-path',None)
        self.load_model_path = params.get('pretrained-model-path',None)
        self.chunk_wise_training = params.get('chunk_wise_training',False)
        self.sequence_length = params.get('sequence_length',99)
        self.n_epochs = params.get('n_epochs', 10)
        self.appliance_params = params.get('appliance_params',{})
        self.mains_mean = params.get('mains_mean',None)
        self.mains_std = params.get('mains_std',None)
        self.batch_size = params.get('batch_size',512)
        self.appliances = params.get('appliances', {})
        self.on_treshold = params.get('on_treshold', 50)

        self.training_history_folder = params.get("training_history_folder", None)
        if self.training_history_folder is not None:
            utils.create_path(self.training_history_folder)

        self.plots_folder = params.get("plots_folder", None)
        if self.plots_folder is not None:
            utils.create_path(self.plots_folder)

    def partial_fit(self, train_main, train_appliances, do_preprocessing=True, current_epoch=0, **load_kwargs):
         # If no appliance wise parameters are provided, then copmute them using the first chunk
        if len(self.appliance_params) == 0:
            self.set_appliance_params(train_appliances)
        if self.mains_mean is None:
            self.set_mains_params(train_main)

        print("...............Seq2Point partial_fit running...............")
        # Do the pre-processing, such as  windowing and normalizing
        if do_preprocessing:
            train_main, train_appliances = self.call_preprocessing(
                train_main, train_appliances, 'train')

        train_main = pd.concat(train_main, axis=0)
        train_main = train_main.values.reshape((-1, self.sequence_length, 1))
        new_train_appliances = []
        for app_name, app_df in train_appliances:
            app_df = pd.concat(app_df, axis=0)
            app_df_values = app_df.values.reshape((-1, 1))
            new_train_appliances.append((app_name, app_df_values))
        train_appliances = new_train_appliances

        for app_name, power in train_appliances:
            
            appliance_model = self.appliances.get(app_name, {})
            transfer_path = appliance_model.get("transfer_path", None)

            if app_name not in self.models:
                if transfer_path is not None:
                    print("Using transfer learning for ", app_name)
                    self.models[app_name] = self.create_transfer_model(transfer_path)
                else:
                    print("First model training for", app_name)
                    self.models[app_name] = self.return_network()
            else:
                print("Started re-training model for", app_name)

            model = self.models[app_name]
            
            filepath = self.file_prefix + "{}.h5".format("_".join(app_name.split()))

            checkpoint = ModelCheckpoint(filepath,monitor='val_loss',verbose=1,save_best_only=True,mode='min')

            app_mean = self.appliance_params[app_name]['mean']
            app_std = self.appliance_params[app_name]['std']

            binary_y = np.array([ 1 if x > self.on_treshold else 0 for x in power*app_std + app_mean])
            
            negatives = np.where(binary_y == 0)[0]
            positives = np.where(binary_y == 1)[0]

            negatives = list(random.sample(set(negatives), positives.shape[0]))
            undersampled_dataset = np.sort(np.concatenate((positives, negatives)))

            train_main = train_main[undersampled_dataset]
            power = power[undersampled_dataset]


            history = model.fit(
                    train_main, power,
                    validation_split=.15,
                    epochs=self.n_epochs,
                    batch_size=self.batch_size,
                    callbacks=[ checkpoint ],
                    shuffle=True,
            )

            model.load_weights(filepath)

            history = json.dumps(history.history)
                    
            if self.training_history_folder is not None:
                f = open(self.training_history_folder + "history_"+app_name.replace(" ", "_")+".json", "w")
                f.write(history)
                f.close()

            if self.plots_folder is not None:
                utils.create_path(self.plots_folder + "/" + app_name + "/")
                plots.plot_model_history_regression(json.loads(history), self.plots_folder + "/" + app_name + "/")

    def disaggregate_chunk(self,test_main_list,model=None,do_preprocessing=True):

        if model is not None:
            self.models = model

        if do_preprocessing:
            test_main_list = self.call_preprocessing(test_main_list, submeters_lst=None, method='test')
        
        test_predictions = []
        for mains in test_main_list:
            disggregation_dict = {}
            mains = mains.values.reshape((-1,self.sequence_length,1))
            for appliance in self.models:
                prediction = self.models[appliance].predict(mains,batch_size=self.batch_size)
                prediction = self.appliance_params[appliance]['mean'] + prediction * self.appliance_params[appliance]['std']
                prediction = np.reshape(prediction, len(prediction))
                valid_predictions = prediction.flatten()
                valid_predictions = np.where(valid_predictions > 0, valid_predictions, 0)
                df = pd.Series(valid_predictions)
                disggregation_dict[appliance] = df
            results = pd.DataFrame(disggregation_dict, dtype='float32')
            test_predictions.append(results)
        return test_predictions

    def save_model(self, save_model_path):
        for appliance_name in self.models:
            print ("Saving model for ", appliance_name)
            self.models[appliance_name].save_weights(os.path.join(save_model_path,appliance_name+".h5"))

    def return_network(self):
        '''Creates the GRU architecture described in the paper
        '''
        model = Sequential()
        # 1D Conv
        model.add(Conv1D(16,4,activation='relu',input_shape=(self.sequence_length,1),padding="same",strides=1))
        # Bi-directional GRUs
        model.add(Bidirectional(GRU(64, activation='relu',
                                    return_sequences=True), merge_mode='concat'))
        model.add(Dropout(0.5))
        model.add(Bidirectional(GRU(128, activation='relu',
                                    return_sequences=False), merge_mode='concat'))
        model.add(Dropout(0.5))
        # Fully Connected Layers
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='linear'))

        model.compile(loss='mean_squared_error', metrics=["MeanAbsoluteError", "RootMeanSquaredError"], optimizer='adam')
        return model

    def create_transfer_model(self, transfer_path):
        model = Sequential()
        # 1D Conv
        model.add(Conv1D(16,4,activation='relu',input_shape=(self.sequence_length,1),padding="same",strides=1))
        # Bi-directional GRUs
        model.add(Bidirectional(GRU(64, activation='relu',
                                    return_sequences=True), merge_mode='concat'))
        model.add(Dropout(0.5))
        model.add(Bidirectional(GRU(128, activation='relu',
                                    return_sequences=False), merge_mode='concat'))
        model.add(Dropout(0.5))

        model.compile(loss='mean_squared_error', metrics=["MeanAbsoluteError", "RootMeanSquaredError"], optimizer='adam')
        model.load_weights(transfer_path, skip_mismatch=True, by_name=True)
        print(model.summary())
        for layer in model.layers:
            layer.trainable = False
        print(model.summary())
        # Fully Connected Layers
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation='linear'))
        print(model.summary())
        model.compile(loss='mean_squared_error', metrics=["MeanAbsoluteError", "RootMeanSquaredError"], optimizer='adam')

        return model

    def call_preprocessing(self, mains_lst, submeters_lst, method):

        if method == 'train':
            # Preprocessing for the train data
            mains_df_list = []
            for mains in mains_lst:
                new_mains = mains.values.flatten()
                n = self.sequence_length
                units_to_pad = n -1
                new_mains = np.pad(new_mains,(units_to_pad, 0),'constant',constant_values=(0,0))
                new_mains = np.array([new_mains[i:i + n] for i in range(len(new_mains) - n + 1)])
                new_mains = (new_mains - self.mains_mean) / self.mains_std
                mains_df_list.append(pd.DataFrame(new_mains))

            appliance_list = []
            for app_name, app_df_list in submeters_lst:
                if app_name in self.appliance_params:
                    app_mean = self.appliance_params[app_name]['mean']
                    app_std = self.appliance_params[app_name]['std']
                else:
                    print ("Parameters for ", app_name ," were not found!")
                    raise ApplianceNotFoundError()

                processed_appliance_dfs = []

                for app_df in app_df_list:
                    new_app_readings = app_df.values.reshape((-1, 1))
                    # This is for choosing windows
                    new_app_readings = (new_app_readings - app_mean) / app_std  
                    # Return as a list of dataframe
                    processed_appliance_dfs.append(pd.DataFrame(new_app_readings))
                appliance_list.append((app_name, processed_appliance_dfs))
            return mains_df_list, appliance_list

        else:
            # Preprocessing for the test data
            mains_df_list = []

            for mains in mains_lst:
                new_mains = mains.values.flatten()
                n = self.sequence_length
                units_to_pad = n -1
                new_mains = np.pad(new_mains,(units_to_pad,0),'constant',constant_values=(0,0))
                new_mains = np.array([new_mains[i:i + n] for i in range(len(new_mains) - n + 1)])
                new_mains = (new_mains - self.mains_mean) / self.mains_std
                mains_df_list.append(pd.DataFrame(new_mains))
            return mains_df_list

    def set_appliance_params(self,train_appliances):

        for (app_name,df_list) in train_appliances:
            l = np.array(pd.concat(df_list,axis=0))
            app_mean = np.mean(l)
            app_std = np.std(l)
            if app_std<1:
                app_std = 100
            self.appliance_params.update({app_name:{'mean':app_mean,'std':app_std}})

    def set_mains_params(self, train_mains):
        l = np.array(pd.concat(train_mains, axis=0))
        self.mains_mean = np.mean(l, axis=0)
        self.mains_std = np.std(l, axis=0)

