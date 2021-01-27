
from nilmtk.disaggregate import Disaggregator
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import matthews_corrcoef, make_scorer
import numpy as np
import pandas as pd

class SVM(Disaggregator):
    def __init__(self):
        self.MODEL_NAME = 'SVM'

    def partial_fit(self, X_train, y_train):

        param = [
            {
                "kernel": ["rbf"],
                "C": [0.03, 0.1, 0.3, 1]
            }
        ]

        svm = SVC()
        mcc_scorer = make_scorer(matthews_corrcoef)
        clf = GridSearchCV(svm, param, cv=5, n_jobs=3, verbose=3, scoring=mcc_scorer)
        clf.fit(X_train, y_train)
        rbf = (clf.best_estimator_, clf.best_score_)

        param = [
            {
                "kernel": ["poly"],
                "degree": [2, 3, 4],
                "C": [0.03, 0.1, 0.3, 1]
            }
        ]
        clf = GridSearchCV(svm, param, cv=5, n_jobs=3, verbose=3, scoring=mcc_scorer)
        clf.fit(X_train, y_train)
        poly = (clf.best_estimator_, clf.best_score_)

        if rbf[1] > poly[1]:
            print(rbf[0])
            self.model = rbf[0]
        else:
            print(poly[0])
            self.model = poly[0]
            
            
    def disaggregate_chunk(self, X_test, y_test):

        pred = self.model.predict(X_test)

        tp = 0
        tn = 0
        fp = 0
        fn = 0
        for i in range(0, len(pred)):
            if pred[i] == y_test[i] and y_test[i] == 1:
                tp +=1
            elif pred[i] == y_test[i] and y_test[i] == 2:
                tn += 1
            elif pred[i] != y_test[i] and y_test[i] == 2:
                fp += 1
            else:
                fn += 1
        
        print("True Positives: ", tp)
        print("True Negatives: ", tn)  
        print("False Negatives: ", fn)  
        print("False Positives: ", fp)        
        print( "MCC: ", matthews_corrcoef(y_test, pred))



    def save_model(self, folder_name):
        #TODO
        return

    def load_model(self, folder_name):
        #TODO
        return
