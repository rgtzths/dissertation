from os.path import join, isdir, isfile
from os import listdir
import re
import numpy as np
import pandas as pd
from svm import SVM
from lstm import LSTM_RNN
from gru import GRU_RNN
from sklearn.utils import shuffle
import math

def test_svm(X_train, y_train, X_test, y_test, app):
    clf = SVM({})
    clf.partial_fit(X_train, y_train, app)
    clf.disaggregate_chunk(X_test, y_test, app)
    clf.save_model(models_folder + "svm")

def test_lstm(X_train, y_train, X_test, y_test, app, model_defenition):
    X_train = X_train.reshape(X_train.shape[0], int(X_train.shape[1]/model_defenition["n_features"]), model_defenition["n_features"])

    X_test = X_test.reshape(X_test.shape[0], int(X_test.shape[1]/model_defenition["n_features"]), model_defenition["n_features"])

    clf = LSTM_RNN( {
        "timeframe": X_train.shape[1], 
        "n_features" : X_train.shape[2], 
        "cv" : model_defenition["cv_percentage"], 
        "epochs": model_defenition["epochs"], 
        "verbose": model_defenition["verbose"] 
        })

    clf.partial_fit(X_train, y_train, app)
    clf.disaggregate_chunk(X_test, y_test, app)
    clf.save_model(models_folder + "lstm")

def test_gru(X_train, y_train, X_test, y_test, app, model_defenition):
    X_train = X_train.reshape(X_train.shape[0], int(X_train.shape[1]/model_defenition["n_features"]), model_defenition["n_features"])

    X_test = X_test.reshape(X_test.shape[0], int(X_test.shape[1]/model_defenition["n_features"]), model_defenition["n_features"])

    clf = GRU_RNN({
        "timeframe": X_train.shape[1], 
        "n_features" : X_train.shape[2], 
        "cv" : model_defenition["cv_percentage"], 
        "epochs": model_defenition["epochs"], 
        "verbose": model_defenition["verbose"]
        })

    clf.partial_fit(X_train, y_train, app)
    clf.disaggregate_chunk(X_test, y_test, app)
    clf.save_model(models_folder + "gru")

################## Valores Editáveis #################

#house_appliances = {"house_1" : ["carcharger", "heatpump"]}

#dataset_folder = "../../datasets/avEiro_timeseries/"

# Mapping between houses and the appliances to be used.
house_appliances = {
    "house_1" : ["boiler", "dish_washer", "fridge", "microwave", "washing_machine"],
    "house_2" : ["dish_washer", "fridge", "kettle", "microwave", "toaster", "washing_machine"],
    "house_3" : ["kettle"],
    "house_5" : ["dish_washer", "fridge", "kettle", "microwave", "oven", "toaster"]
}

#Dataset folder
dataset_folder = "../../datasets/ukdale_timeseries/"

#Folder to save the models
models_folder = "./models/"

#Amount of data to be used during training 
train_size = 0.7

#Time elapsed between feature vectors
timestep = 5

data_for_DNN = {
    "carcharger" : { "n_samples": 0, "cv_percentage": 0.16, "n_features" : 2, "epochs": 300, "verbose": 0},
    "heatpump" : { "n_samples": 0, "cv_percentage": 0.16, "n_features" : 2, "epochs": 300, "verbose": 0},
    "boiler" : { "n_samples": 0, "cv_percentage": 0.16, "n_features" : 1, "epochs": 300, "verbose": 0},
    "dish_washer" : { "n_samples": 0, "cv_percentage": 0.16, "n_features" : 1, "epochs": 300, "verbose": 0},
    "fridge" : { "n_samples": 0, "cv_percentage": 0.16, "n_features" : 1, "epochs": 300, "verbose": 0},
    "microwave" : { "n_samples": 0, "cv_percentage": 0.16, "n_features" : 1, "epochs": 300, "verbose": 0},
    "washing_machine" : { "n_samples": 0, "cv_percentage": 0.16, "n_features" : 1, "epochs": 300, "verbose": 0},
    "kettle" : { "n_samples": 0, "cv_percentage": 0.16, "n_features" : 1, "epochs": 300, "verbose": 0},
    "toaster" : { "n_samples": 0, "cv_percentage": 0.16, "n_features" : 1, "epochs": 300, "verbose": 0},
    "oven" : { "n_samples": 0, "cv_percentage": 0.16, "n_features" : 1, "epochs": 300, "verbose": 0}
}

######################################################################################
#Array that contains the aggragate readings in dataframes
aggregated_readings = []
#Dictionary that contains the readings from the appliances
# With format { "app" : {"house" : dataframe}}
app_readings = {}

#Goes through all the houses
for house in house_appliances.keys():
    #Loads the mains files
    df = pd.read_csv(dataset_folder+house+"/mains.csv", sep=',', header=None)
    #converts the times to date time and uses it as index
    df.index = pd.to_datetime(df[0])
    #drops the column that had the time
    df = df.drop(0, 1)
    #appends that dataframe to an array
    aggregated_readings.append(df)

    #Loads the appliances to be used from that house using a similar logic.
    for app in house_appliances[house]:
        df = pd.read_csv(dataset_folder+house+ "/"+ app + ".csv", sep=',', header=None, names=["time", "value"])
        df.index = pd.to_datetime(df["time"])
        df = df.drop("time", 1)
        #Stores the dataframe in a dictionary
        if app in app_readings:
            app_readings[app][house] = df
        else:
            app_readings[app] = {house : df}

#For each appliance loads the data and does the training/test.
for app in app_readings.keys():
    #Variables that contain X and y
    X = []
    y = []

    #Goes through all the houses.
    for i, house in enumerate(list(house_appliances.keys())):
        #Checks if that house is to be used on this appliance
        if app in house_appliances[house]:
            #Intersects the intervals between the mains data and the appliance data.
            if aggregated_readings[i].index[0] < app_readings[app][house].index[0]:
                beggining = app_readings[app][house].index[0].timestamp()
            else: 
                beggining = aggregated_readings[i].index[0].timestamp()
            
            if aggregated_readings[i].index[-1] < app_readings[app][house].index[-1]:
                end = aggregated_readings[i].index[-1].timestamp()
            else:
                end = app_readings[app][house].index[-1].timestamp()
            
            beggining_index_x = round((beggining - aggregated_readings[i].index[0].timestamp()) / (60*timestep))

            if aggregated_readings[i].index[-1].timestamp() != end:
                end_index_x = len(aggregated_readings[i]) - round((aggregated_readings[i].index[-1].timestamp() - end)/ (60*timestep)) -1
            else:
                end_index_x = -1

            beggining_index_y = round((beggining - app_readings[app][house].index[0].timestamp()) / (60*timestep))

            if app_readings[app][house].index[-1].timestamp() != end:
                end_index_y =  len(app_readings[app][house]) - round((app_readings[app][house].index[-1].timestamp() - end)/ (60*timestep)) - 1
            else:
                end_index_y = -1

            #Appends to the X and Y the intersection of the mains and appliance readings
            X.append(aggregated_readings[i][beggining_index_x: end_index_x])

            y.append(app_readings[app][house][beggining_index_y: end_index_y])

    #Using concat appends the X and y values using the rows
    X = pd.concat(X, axis=0).values
    y = pd.concat(y, axis=0).values
    
    #Shuffles the values
    X, y = shuffle(X, y, random_state=0)
    
    #Splits in training and testing.
    X_train = X[0:int(len(X)*train_size)]
    y_train = y[0:int(len(X)*train_size)]
    X_test = X[int(len(X)*train_size):]
    y_test = y[int(len(X)*train_size):]
    
    #Does the training and testing.
    test_lstm(X_train, y_train, X_test, y_test, app, data_for_DNN[app])
    test_gru(X_train, y_train, X_test, y_test, app, data_for_DNN[app])
    test_svm(X_train, y_train, X_test, y_test, app)