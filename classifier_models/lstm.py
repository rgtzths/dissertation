import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=RuntimeWarning)

from os.path import join, isfile
from os import listdir
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
import keras.backend as K
from sklearn.metrics import matthews_corrcoef, confusion_matrix
import numpy as np
import dataset_loader


def matthews_correlation(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())

class LSTM_RNN():
    def __init__(self, params):
        self.model = {}
        self.MODEL_NAME = params.get('model_name', 'LSTM RNN')
        self.timeframe = params.get('timeframe', 10)
        self.timestep = params.get('timestep', 2)
        self.overlap = params.get('overlap', 0.5)
        self.interpolate = params.get('interpolate', 'average')
        self.cv = params.get('cv', 0.16)
        self.load_model_path = params.get('load_model_folder',None)
        self.input_size = params.get('input_size', 0)
        self.epochs = params.get('epochs', 300)
        self.verbose = params.get('verbose', 0)

        if self.load_model_path:
            self.load_model(self.load_model_path)

    def partial_fit(self, train_main, train_appliance, app):
        X_train = dataset_loader.generate_main_timeseries(train_main, self.timeframe, self.timestep, self.overlap, self.interpolate)
        y_train = dataset_loader.generate_appliance_timeseries(train_appliance, self.timeframe, self.timestep, self.overlap, self.interpolate)
        X_train = X_train.reshape(X_train.shape[0], int(self.timeframe*60/self.timestep), len(train_main[0].columns.values))

        X_cv = X_train[int(len(X_train)*(1-self.cv)):]
        y_cv = y_train[int(len(y_train)*(1-self.cv)):]

        X_train = X_train[0:int(len(X_train)*(1-self.cv))]
        y_train = y_train[0:int(len(y_train)*(1-self.cv))]

        if app in self.model:
            model = self.model[app]
        else:
            model = Sequential()
            model.add(LSTM(X_train.shape[1] + X_train.shape[1]*self.input_size, input_shape=(X_train.shape[1], X_train.shape[2])))
            model.add(Dense(units=1, activation='sigmoid'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[matthews_correlation])
        
        model.fit(X_train, y_train, epochs=self.epochs, batch_size=1000, validation_data=(X_cv, y_cv), verbose=self.verbose, shuffle=False)

        del X_train
        del y_train
        del X_cv
        del y_cv

        self.model[app] = model

    def disaggregate_chunk(self, test_main, test_appliance, app):

        X_test = dataset_loader.generate_main_timeseries(test_main, self.timeframe, self.timestep, self.overlap, self.interpolate)
        y_test = dataset_loader.generate_appliance_timeseries(test_appliance, self.timeframe, self.timestep, self.overlap, self.interpolate)

        X_test = X_test.reshape(X_test.shape[0], int(self.timeframe*60/self.timestep), len(test_main[0].columns.values))
        
        pred = self.model[app].predict(X_test)
        pred = [ 0 if p < 0.5 else 1 for p in pred ]

        y_test = y_test.reshape(len(y_test),)

        tn, fn, fp, tp = confusion_matrix(y_test, pred).ravel()
        
        mcc = matthews_corrcoef(y_test, pred)

        print("True Positives: ", tp)
        print("True Negatives: ", tn)  
        print("False Negatives: ", fn)  
        print("False Positives: ", fp)        
        print( "MCC: ", mcc)

        del X_test
        del y_test

        return mcc

    def save_model(self, folder_name):
        for app in self.model:
            self.model[app].save(join(folder_name, app + "_" + self.MODEL_NAME + ".h5"))

    def load_model(self, folder_name):
        app_models = [f for f in listdir(folder_name) if isfile(join(folder_name, f))]
        for app in app_models:
            self.model[app.split(".")[0]] = load_model(join(folder_name, app))