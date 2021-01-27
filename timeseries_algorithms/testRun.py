



import numpy as np
import pandas as pd
from SVM import SVM
from sklearn.utils import shuffle

agregated_readings = "../converters/avEiro_timeseries/house_1/channel_1.csv"
appliance_status = "../converters/avEiro_timeseries/house_1/channel_2.csv"

train_size = 0.7

X = pd.read_csv(agregated_readings, sep=',', header=None)

Y = pd.read_csv(appliance_status, sep=',', header=None) 

y = list(Y[0])

X, y = shuffle(X, y, random_state=0)

X_train = X[0:int(len(X)*train_size)]
y_train = y[0:int(len(X)*train_size)]

X_test = X[int(len(X)*train_size):]
y_test = y[int(len(X)*train_size):]

clf = SVM()

clf.partial_fit(X_train, y_train)
clf.disaggregate_chunk(X_test,y_test)

