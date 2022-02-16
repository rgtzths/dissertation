from collections import OrderedDict
import numpy as np
import pandas as pd
from nilmtk.disaggregate import Disaggregator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Reshape, Flatten
from tensorflow.keras.models import Sequential
import os, json

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
        self.file_prefix = "{}-temp-weights".format(self.MODEL_NAME.lower())
        self.chunk_wise_training = params.get('chunk_wise_training',False)
        self.sequence_length = params.get('sequence_length',99)
        self.n_epochs = params.get('n_epochs', 10 )
        self.batch_size = params.get('batch_size',512)
        self.appliance_params = params.get('appliance_params',{})
        self.mains_mean = params.get('mains_mean',1800)
        self.mains_std = params.get('mains_std',600)
        if self.sequence_length%2==0:
            print ("Sequence length should be odd!")
            raise (SequenceLengthError)

    def partial_fit(self, train_data, cv_data=None):
        # If no appliance wise parameters are provided, then copmute them using the first chunk
        if len(self.appliance_params) == 0:
            self.set_appliance_params(train_data)

        print("...............Seq2Point partial_fit running...............")

        for appliance_name, data in train_data.items():

            train_main, train_appliance = self.call_preprocessing(data["mains"], data["appliance"], appliance_name, 'train')

            train_main = pd.concat(train_main, axis=0).values.reshape((-1, self.sequence_length, 1))
            train_appliance = pd.concat(train_appliance, axis=0).values.reshape((-1, 1))
            
            if cv_data is not None:
                cv_main, cv_appliance = self.call_preprocessing(cv_data[appliance_name]["mains"], cv_data[appliance_name]["appliance"], appliance_name, 'train')
                
                cv_main = pd.concat(cv_main, axis=0).values.reshape((-1,self.sequence_length,1))
                cv_appliance = pd.concat(cv_appliance, axis=0).values.reshape((-1, 1))

            # Check if the appliance was already trained. If not then create a new model for it
            if appliance_name not in self.models:
                print("First model training for", appliance_name)
                self.models[appliance_name] = self.return_network()
            # Retrain the particular appliance
            else:
                print("Started Retraining model for", appliance_name)

            model = self.models[appliance_name]

            filepath = self.file_prefix + "-{}-epoch{}.h5".format(
                    "_".join(appliance_name.split()),
                    0,
            )
            checkpoint = ModelCheckpoint(filepath,monitor='val_loss',verbose=1,save_best_only=True,mode='min')
            if cv_data is not None:
                model.fit(
                    train_main, train_appliance,
                    validation_data=(cv_main, cv_appliance),
                    epochs=self.n_epochs,
                    batch_size=self.batch_size,
                    callbacks=[checkpoint],
                    shuffle=False,
                )
            else:
                model.fit(
                    train_main, train_appliance,
                    validation_split=.15,
                    epochs=self.n_epochs,
                    batch_size=self.batch_size,
                    callbacks=[checkpoint],
                    shuffle=False,
                )
            model.load_weights(filepath)

                    
    def disaggregate_chunk(self, test_mains, appliance):

        test_main_list = self.call_preprocessing(test_mains, app_df_list=None, app_name=appliance, method='test')

        test_predictions = []
        for test_main in test_main_list:
            test_main = test_main.values
            test_main = test_main.reshape((-1, self.sequence_length, 1))
            disggregation_dict = {}
            prediction = self.models[appliance].predict(test_main,batch_size=self.batch_size)
            prediction = self.appliance_params[appliance]['mean'] + prediction * self.appliance_params[appliance]['std']
            valid_predictions = prediction.flatten()
            valid_predictions = np.where(valid_predictions > 0, valid_predictions, 0)
            df = pd.Series(valid_predictions)
            disggregation_dict[appliance] = df
            results = pd.DataFrame(disggregation_dict, dtype='float32')
            test_predictions.append(results)
        return test_predictions

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
        model.compile(loss='mse', optimizer='adam')  # ,metrics=[self.mse])
        return model

    def call_preprocessing(self, mains_lst, app_df_list, app_name, method):

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

                app_mean = self.appliance_params[app_name]['mean']
                app_std = self.appliance_params[app_name]['std']

                processed_appliance_dfs = []

                for app_df in app_df_list:
                    new_app_readings = app_df.values.reshape((-1, 1))
                    # This is for choosing windows
                    new_app_readings = (new_app_readings - app_mean) / app_std  
                    # Return as a list of dataframe
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
            self.appliance_params.update({app_name:{'mean':float(app_mean),'std':float(app_std)}})

    def load_model(self, path):
        return True

    def save_model(self, path):
        params_to_save = {}
        params_to_save['appliance_params'] = self.appliance_params
        params_to_save['sequence_length'] = self.sequence_length
        params_to_save['mains_mean'] = self.mains_mean
        params_to_save['mains_std'] = self.mains_std
        for appliance_name in self.models:
            print ("Saving model for ", appliance_name)
            self.models[appliance_name].save_weights(os.path.join(path,appliance_name+".h5"))

        with open(os.path.join(path,appliance_name+'.json'),'w') as file:
            file.write(json.dumps(params_to_save))