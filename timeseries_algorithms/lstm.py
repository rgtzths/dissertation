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
        self.MODEL_NAME = 'LSTM RNN'
        self.model = {}
        self.timeframe = params.get('timeframe', None)
        self.n_features = params.get('n_features', None)
        self.cv = params.get('cv', 0.16)
        self.load_model_path = params.get('load_model_folder',None)
        self.input_size = params.get('input_size', 0)
        self.epochs = params.get('epochs', 300)
        self.verbose = params.get('verbose', 0)
        if self.load_model_path:
            self.load_model(self.load_model_path)

    def partial_fit(self, X_train, y_train, app):
        X_cv = X_train[int(len(X_train)*(1-self.cv)):]
        y_cv = y_train[int(len(y_train)*(1-self.cv)):]

        X_train = X_train[0:int(len(X_train)*(1-self.cv))]
        y_train = y_train[0:int(len(y_train)*(1-self.cv))]

        if app in self.model:
            model = self.model[app]
        else:
            model = Sequential()
            model.add(LSTM(X_train.shape[1] + X_train.shape[1]*self.input_size, input_shape=(self.timeframe, self.n_features)))
            model.add(Dense(units=1, activation='sigmoid'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[matthews_correlation])
        
        model.fit(X_train, y_train, epochs=self.epochs, batch_size=1000, validation_data=(X_cv, y_cv), verbose=self.verbose, shuffle=False)

        self.model[app] = model
            
    def disaggregate_chunk(self, X_test, y_test, app):
        pred = self.model[app].predict(X_test)
        pred = [ 0 if p < 0.5 else 1 for p in pred ]
        y_test = y_test.reshape(len(y_test),)

        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        
        print("True Positives: ", tp)
        print("True Negatives: ", tn)  
        print("False Negatives: ", fn)  
        print("False Positives: ", fp)        
        print( "MCC: ", matthews_corrcoef(y_test, pred))

    def save_model(self, folder_name):
        for app in self.model:
            self.model[app].save(join(folder_name, app + ".h5"))

    def load_model(self, folder_name):
        app_models = [f for f in listdir(folder_name) if isfile(join(folder_name, f))]
        for app in app_models:
            self.model[app.split(".")[0]] = load_model(join(folder_name, app))