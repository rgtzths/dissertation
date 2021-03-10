import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=RuntimeWarning)

from os.path import join, isfile
from os import listdir
from keras.models import Sequential, load_model
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
import keras.backend as K
from sklearn.metrics import matthews_corrcoef, confusion_matrix
import pywt
import numpy as np

import sys
sys.path.insert(1, "../feature_extractors")
import dataset_loader
import generate_timeseries

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

class CNN():
    def __init__(self, params):
        self.MODEL_NAME = params.get('model_name', 'CNN')
        self.model = {}
        self.timeframe = params.get('timeframe', None)
        self.timestep = params.get('timestep', 2)
        self.overlap = params.get('overlap', 0.5)
        self.interpolate = params.get('interpolate', 'average')
        self.column = params.get('predicted_column', ("power", "apparent"))
        self.cv = params.get('cv', 0.16)
        self.load_model_path = params.get('load_model_folder',None)
        self.epochs = params.get('epochs', 300)
        self.verbose = params.get('verbose', 0)
        self.waveletname = params.get('waveletname', 'morl')
        
        if self.load_model_path:
            self.load_model(self.load_model_path)

    def partial_fit(self, train_main, train_appliance, app):

        train_data = generate_timeseries.generate_main_timeseries(train_main, False, self.timeframe, self.timestep, self.overlap, self.interpolate)
        y_train = generate_timeseries.generate_appliance_timeseries(train_appliance, True, self.timeframe, self.timestep, self.overlap, self.column, self.interpolate)

        train_data = train_data.reshape(train_data.shape[0], int(self.timeframe*60/self.timestep), len(train_main[0].columns.values))
        
        X_train = np.ndarray(shape=(train_data.shape[0], train_data.shape[1]-1, train_data.shape[1]-1, train_data.shape[2]))

        for i in range(0, train_data.shape[0]):
            for j in range(0, train_data.shape[2]):
                signal = train_data[i, :, j]
                coeff, freq = pywt.cwt(signal, range(1, train_data.shape[1]), self.waveletname, 1)
                coeff_ = coeff[:,:train_data.shape[1]-1]
                X_train[i, :, :, j] = coeff_
        
        X_cv = X_train[int(len(X_train)*(1-self.cv)):]
        y_cv = y_train[int(len(y_train)*(1-self.cv)):]

        X_train = X_train[0:int(len(X_train)*(1-self.cv))]
        y_train = y_train[0:int(len(y_train)*(1-self.cv))]
        
        if app in self.model:
            model = self.model[app]
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
            model.add(Dense(units=1, activation='sigmoid'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[matthews_correlation])

        model.fit(X_train, y_train, epochs=self.epochs, batch_size=1000, validation_data=(X_cv, y_cv), verbose=self.verbose, shuffle=False)

        del X_train
        del y_train
        del X_cv
        del y_cv
        
        self.model[app] = model
            
    def disaggregate_chunk(self, test_main, test_appliance, app):
        test_data = generate_timeseries.generate_main_timeseries(test_main, False, self.timeframe, self.timestep, self.overlap, self.interpolate)
        y_test = generate_timeseries.generate_appliance_timeseries(test_appliance, True, self.timeframe, self.timestep, self.overlap, self.column, self.interpolate)

        test_data = test_data.reshape(test_data.shape[0], int(self.timeframe*60/self.timestep), len(test_main[0].columns.values))

        X_test = np.ndarray(shape=(test_data.shape[0], test_data.shape[1]-1, test_data.shape[1]-1, test_data.shape[2]))

        for i in range(0, test_data.shape[0]):
            for j in range(0, test_data.shape[2]):
                signal = test_data[i, :, j]
                coeff, freq = pywt.cwt(signal, range(1, test_data.shape[1]), self.waveletname, 1)
                coeff_ = coeff[:,:test_data.shape[1]-1]
                X_test[i, :, :, j] = coeff_ 


        pred = self.model[app].predict(X_test)
        pred = [ 0 if p < 0.5 else 1 for p in pred ]
        
        y_test = y_test.reshape(len(y_test),)
        
        #tn, fn, fp, tp = confusion_matrix(y_test, pred).ravel()
        
        #print("True Positives: ", tp)
        #print("True Negatives: ", tn)  
        #print("False Negatives: ", fn)  
        #print("False Positives: ", fp)        
        #print( "MCC: ", matthews_corrcoef(y_test, pred))

        return matthews_corrcoef(y_test, pred)

    def save_model(self, folder_name):
        for app in self.model:
            self.model[app].save(join(folder_name, app + "_" + self.MODEL_NAME + ".h5"))

    def load_model(self, folder_name):
        app_models = [f for f in listdir(folder_name) if isfile(join(folder_name, f))]
        for app in app_models:
            self.model[app.split(".")[0]] = load_model(join(folder_name, app))
