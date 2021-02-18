
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import matthews_corrcoef, make_scorer, confusion_matrix
import numpy as np
import pandas as pd
import joblib

class SVM():
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

    def partial_fit(self, X_train, y_train, app):

        if app in self.model:
            clf = self.model[app]
        else:
            clf = SVC(kernel=self.kernel, C=self.C, epsilon=self.epsilon, coef0=self.coef, degree=self.degree, tol=self.tol)
        
        clf.fit(X_train, y_train)
        
        self.model[app] = clf
            
            
    def disaggregate_chunk(self, X_test, y_test, app):

        pred = self.model[app].predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
        
        print("True Positives: ", tp)
        print("True Negatives: ", tn)  
        print("False Negatives: ", fn)  
        print("False Positives: ", fp)        
        print( "MCC: ", matthews_corrcoef(y_test, pred))

    def save_model(self, folder_name):
        for app in self.model:
            joblib.dump(self.model[app], join(folder_name, app+".sav"))

    def load_model(self, folder_name):
        app_models = [f for f in listdir(folder_name) if isfile(join(folder_name, f))]
        for app in app_models:
            self.model[app.split(".")[0]] = joblib.load(join(folder_name, app))
        
