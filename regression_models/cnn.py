import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=RuntimeWarning)

from os.path import join, isfile
from os import listdir

from nilmtk.disaggregate import Disaggregator
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from keras.callbacks import History
import pandas as pd
import pywt
import numpy as np
from generate_timeseries import generate_main_timeseries, generate_appliance_timeseries

class CNN(Disaggregator):
    def __init__(self, params):
        self.model = {}
        self.MODEL_NAME = params.get('model_name', 'CNN')
        self.timeframe = params.get('timeframe', 5)
        self.timestep = params.get('timestep', 2)
        self.overlap = params.get('overlap', 0.5)
        self.interpolate = params.get('interpolate', 'average')
        self.cv = params.get('cv', 0.16)
        self.column = params.get('predicted_column', ("power", "apparent"))
        self.load_model_path = params.get('load_model_folder',None)
        self.epochs = params.get('epochs', 300)
        self.verbose = params.get('verbose', 0)
        self.waveletname = params.get('waveletname', 'morl')

        if self.load_model_path:
            self.load_model(self.load_model_path)

    def partial_fit(self, train_main, train_appliances, **load_kwargs):

        print("Preparing the Training Data: X")

        train_data = generate_main_timeseries(train_main, False, self.timeframe, self.overlap, self.timestep, self.interpolate)

        train_data = train_data.reshape(train_data.shape[0], int(self.timeframe*60/self.timestep), len(train_main[0].columns.values))
        
        X_train = np.ndarray(shape=(train_data.shape[0], train_data.shape[1]-1, train_data.shape[1]-1, train_data.shape[2]))

        for i in range(0, train_data.shape[0]):
            for j in range(0, train_data.shape[2]):
                signal = train_data[i, :, j]
                coeff, freq = pywt.cwt(signal, range(1, train_data.shape[1]), self.waveletname, 1)
                coeff_ = coeff[:,:train_data.shape[1]-1]
                X_train[i, :, :, j] = coeff_        
        
        X_cv = X_train[int(len(X_train)*(1-self.cv)):]
        
        X_train = X_train[0:int(len(X_train)*(1-self.cv))]
        for app_name, power in train_appliances:
            print("Preparing the Training Data: Y")
            y_train = generate_appliance_timeseries(power, self.timeframe, self.overlap, self.timestep, self.column, self.interpolate)
            
            y_cv = y_train[int(len(y_train)*(1-self.cv)):]
            y_train = y_train[0:int(len(y_train)*(1-self.cv))]
           
            print("Training ", app_name, " in ", self.MODEL_NAME, " model\n", end="\r")

            if app_name in self.model:
                model = self.model[app_name]
            else:
                model = Sequential()
                model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                                activation='relu',
                                input_shape=(X_train.shape[1], X_train.shape[1], X_train.shape[3])))

                model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
                model.add(Conv2D(64, (5, 5), activation='relu'))
                model.add(MaxPooling2D(pool_size=(2, 2)))
                model.add(Flatten())
                model.add(Dense(1000, activation='relu'))
                model.add(Dense(1))
                model.compile(optimizer='adam', loss='mean_squared_error', metrics=["RootMeanSquaredError"])

            model.fit(X_train, y_train, epochs=self.epochs, batch_size=1000, validation_data=(X_cv, y_cv), verbose=self.verbose, shuffle=True)

            self.model[app_name] = model
            
    def disaggregate_chunk(self, test_mains):
        test_predictions_list = []

        print("Preparing the Test Data")
        test_data = generate_main_timeseries(test_mains, True, self.timeframe, self.overlap, self.timestep, self.interpolate)
        
        test_data = test_data.reshape(test_data.shape[0], int(self.timeframe*60/self.timestep), len(test_mains[0].columns.values))
        
        X_test = np.ndarray(shape=(test_data.shape[0], test_data.shape[1]-1, test_data.shape[1]-1, test_data.shape[2]))

        for i in range(0, test_data.shape[0]):
            for j in range(0, test_data.shape[2]):
                signal = test_data[i, :, j]
                coeff, freq = pywt.cwt(signal, range(1, test_data.shape[1]), self.waveletname, 1)
                coeff_ = coeff[:,:test_data.shape[1]-1]
                X_test[i, :, :, j] = coeff_     

        appliance_powers_dict = {}

        for i, app_name in enumerate(self.model):
        
            print("Estimating power demand for '{}' in '{}'\n".format(app_name, self.MODEL_NAME))
            pred = self.model[app_name].predict(X_test)
            pred = [p[0] for p in pred]

            column = pd.Series(
                    pred, index=test_mains[0].index, name=i)
            appliance_powers_dict[app_name] = column
        
        test_predictions_list.append(pd.DataFrame(appliance_powers_dict, dtype='float32'))

        return test_predictions_list

    def save_model(self, folder_name):
        for app in self.model:
            self.model[app].save(join(folder_name, app + "_" + self.MODEL_NAME + ".h5"))

    def load_model(self, folder_name):
        app_models = [f for f in listdir(folder_name) if isfile(join(folder_name, f))]
        for app in app_models:
            self.model[app.split(".")[0]] = load_model(join(folder_name, app))
