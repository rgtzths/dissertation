from collections import OrderedDict
import numpy as np
import pandas as pd
from nilmtk.disaggregate import Disaggregator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Conv1D, Dense, Dropout, Reshape, Flatten
from tensorflow.keras.models import Sequential

from sklearn.model_selection import train_test_split
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
        self.file_prefix = "{}-temp-weights".format(params.get('file_prefix', "") + self.MODEL_NAME.lower())
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

        for appliance_name, power in train_appliances:
            # Check if the appliance was already trained. If not then create a new model for it
            if appliance_name not in self.models:
                print("First model training for", appliance_name)
                self.models[appliance_name] = self.return_network()
            # Retrain the particular appliance
            else:
                print("Started Retraining model for", appliance_name)

            model = self.models[appliance_name]
            if train_main.size > 0:
                # Sometimes chunks can be empty after dropping NANS
                if len(train_main) > 10:
                    # Do validation when you have sufficient samples
                    filepath = self.file_prefix + "-{}-epoch{}.h5".format(
                            "_".join(appliance_name.split()),
                            current_epoch,
                    )
                    checkpoint = ModelCheckpoint(filepath,monitor='val_loss',verbose=1,save_best_only=True,mode='min')
                    X_train, X_cv, y_train, y_cv = train_test_split(train_main, power, test_size=.15, stratify=[ 1 if x > 80 else 0 for x in power])

                    history = model.fit(
                            X_train, y_train,
                            validation_data=(X_cv, y_cv),
                            epochs=self.n_epochs,
                            batch_size=self.batch_size,
                            callbacks=[checkpoint],
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

        # Preprocess the test mains such as windowing and normalizing

        if do_preprocessing:
            test_main_list = self.call_preprocessing(test_main_list, submeters_lst=None, method='test')

        test_predictions = []
        for test_main in test_main_list:
            test_main = test_main.values
            test_main = test_main.reshape((-1, self.sequence_length, 1))
            disggregation_dict = {}
            for appliance in self.models:
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
        model.compile(loss='mean_squared_error', metrics=["MeanAbsoluteError", "RootMeanSquaredError"], optimizer='adam')
        return model

    def call_preprocessing(self, mains_lst, submeters_lst, method):

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

            appliance_list = []
            for app_index, (app_name, app_df_list) in enumerate(submeters_lst):
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
                units_to_pad = n // 2
                new_mains = np.pad(new_mains,(units_to_pad,units_to_pad),'constant',constant_values=(0,0))
                new_mains = np.array([new_mains[i:i + n] for i in range(len(new_mains) - n + 1)])
                new_mains = (new_mains - self.mains_mean) / self.mains_std
                mains_df_list.append(pd.DataFrame(new_mains))
            return mains_df_list

    def set_appliance_params(self,train_appliances):
        # Find the parameters using the first
        for (app_name,df_list) in train_appliances:
            l = np.array(pd.concat(df_list,axis=0))
            app_mean = np.mean(l)
            app_std = np.std(l)
            if app_std<1:
                app_std = 100
            self.appliance_params.update({app_name:{'mean':app_mean,'std':app_std}})
        print (self.appliance_params)
