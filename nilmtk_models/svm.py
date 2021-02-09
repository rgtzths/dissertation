
from nilmtk.disaggregate import Disaggregator
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
import numpy as np
import pandas as pd

class Svm(Disaggregator):
    def __init__(self, params):
        self.model = {}
        self.MODEL_NAME = 'SVM'

    def partial_fit(self, train_main, train_appliances, **load_kwargs):
        x_train = train_main[0].values   
        x_train = x_train.reshape( x_train.shape[0], len(train_main[0].columns.values))

        for app_name, power in train_appliances:
            print("Training ", app_name, " in ", self.MODEL_NAME, " model\n", end="\r")
            
            y_train = power[0]["power"]["apparent"].values
            clf = SVR(kernel="rbf", C=0.1)
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
        string_to_save = json.dumps(self.model)
        os.makedirs(folder_name, exist_ok=True)
        with open(os.path.join(folder_name, "model.txt"), "w") as f:
            f.write(string_to_save)

    def load_model(self, folder_name):
        with open(os.path.join(folder_name, "model.txt"), "r") as f:
            model_string = f.read().strip()
            self.model = json.loads(model_string)
