



import numpy as np
import pandas as pd
from svm import SVM
from lstm import LSTM_RNN
from gru import GRU_RNN
from sklearn.utils import shuffle

agregated_readings = "../../datasets/avEiro_timeseries/house_1/channel_1.csv"
appliance_status = "../../datasets/avEiro_timeseries/house_1/channel_2.csv"

train_size = 0.7

#Data for RNN
n_samples = 300
n_features = 1
cv_percentage = 0.16


X = pd.read_csv(agregated_readings, sep=',', header=None).values.astype('float32')

y = pd.read_csv(appliance_status, sep=',', header=None).values.astype('float32')

X, y = shuffle(X, y, random_state=0)

X_train = X[0:int(len(X)*train_size)]
y_train = y[0:int(len(X)*train_size)]

X_test = X[int(len(X)*train_size):]
y_test = y[int(len(X)*train_size):]

def test_svm(X_train, y_train, X_test, y_test, app):

    clf = SVM({})
    clf.partial_fit(X_train, y_train, app)
    clf.disaggregate_chunk(X_test, y_test, app)

def test_lstm(X_train, y_train, X_test, y_test, app):
    X_train = X_train.reshape(X_train.shape[0], n_samples, n_features)

    X_test = X_test.reshape(X_test.shape[0], n_samples, n_features)

    clf = LSTM_RNN( {"timeframe": X_train.shape[1], "n_features" : X_train.shape[2], "cv" : cv_percentage })

    clf.partial_fit(X_train, y_train, app)
    clf.disaggregate_chunk(X_test, y_test, app)

def test_gru(X_train, y_train, X_test, y_test, app):
    X_train = X_train.reshape(X_train.shape[0], n_samples, n_features)

    X_test = X_test.reshape(X_test.shape[0], n_samples, n_features)

    clf = GRU_RNN({"timeframe": X_train.shape[1], "n_features" : X_train.shape[2], "cv" : cv_percentage})

    clf.partial_fit(X_train, y_train, app)
    clf.disaggregate_chunk(X_test, y_test, app)

test_lstm(X_train, y_train, X_test, y_test)
test_gru(X_train, y_train, X_test, y_test)