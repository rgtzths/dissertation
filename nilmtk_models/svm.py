
from nilmtk.disaggregate import Disaggregator
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd
from os.path import isfile, join
from os import listdir
import joblib

class Svm(Disaggregator):
    def __init__(self, params):
        self.model = {}
        self.MODEL_NAME = 'SVM'
        self.kernel = params.get('kernel', "rbf")
        self.C = params.get('C', "0.1")
        self.degree = params.get('degree', 1)
        self.coef = params.get('coef', 0.1)
        self.epsilon = params.get("epsilon", 0.1)
        self.tol = params.get("tol", 0.0001)
        self.save_model_path = params.get('save_model_folder', None)
        self.load_model_path = params.get('load_model_folder',None)
        if self.load_model_path:
            self.load_model(self.load_model_path)


    def partial_fit(self, train_main, train_appliances, **load_kwargs):
        x_train = train_main[0].values
        x_train = x_train.reshape( x_train.shape[0], len(train_main[0].columns.value) )

        for app_name, power in train_appliances:
            print("Training ", app_name, " in ", self.MODEL_NAME, " model\n", end="\r")
            
            y_train = power[0]["power"]["apparent"].values

            if app_name in self.model:
                clf = self.model[app_name]
            else:
                clf = SVR(kernel=self.kernel, C=self.C, epsilon=self.epsilon, coef0=self.coef, degree=self.degree, tol=self.tol)
            
            clf.fit(x_train, y_train)
            self.model[app_name] = clf
                        
    def disaggregate_chunk(self, test_mains):
        test_predictions_list = []
        x_test = test_mains[0].values
        x_test = x_test.reshape( x_test.shape[0], len(test_mains.columns.value))

        appliance_powers_dict = {}

        for i, app_name in enumerate(self.model):

            print("Estimating power demand for '{}' in '{}'\n".format(app_name, self.MODEL_NAME))
            pred = self.model[app_name].predict(x_test)
            
            column = pd.Series(
                    pred, index=test_mains[0].index, name=i)
            appliance_powers_dict[app_name] = column
            
        appliance_powers = pd.DataFrame(
                appliance_powers_dict, dtype='float32')
        test_predictions_list.append(appliance_powers)

        return test_predictions_list

    def save_model(self, folder_name):
        for app in self.model:
            joblib.dump(self.model[app], join(folder_name, app+".sav"))


    def load_model(self, folder_name):
        app_models = [f for f in listdir(folder_name) if isfile(join(folder_name, f))]
        for app in app_models:
            self.model[app.split(".")[0]] = joblib.load(join(folder_name, app))
        