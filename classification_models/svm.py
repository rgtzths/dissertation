import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=RuntimeWarning)

from sklearn.svm import SVC

from sklearn.metrics import matthews_corrcoef, confusion_matrix

import numpy as np

#import sys
#sys.path.insert(1, "../utils")
#sys.path.insert(1, "../feature_extractors")

from generate_timeseries import generate_main_timeseries, generate_appliance_timeseries

import utils

class SVM():
    def __init__(self, params):
        #Variable that will store the models trained for each appliance.
        self.model = {}
        #Name used to identify the Model Name.
        self.MODEL_NAME = params.get('model_name', 'SVM')
        #Percentage of values used as cross validation data from the training data.
        self.cv_split = params.get('cv_split', 0.16)
        #If this variable is not None than the class loads the appliance models present in the folder.
        self.load_model_path = params.get('load_model_path',None)
        #Dictates the ammount of information to be presented during training and regression.
        self.verbose = params.get('verbose', 1)

        self.appliances = params["appliances"]

        self.random_search = params.get("random_search", False)

        self.default_appliance = {
            "timewindow": 180,
            "timestep": 2,
            "overlap": 178,
            'epochs' : 300,
            'batch_size' : 1024,
            "on_treshold" : 50
        }

        self.results_folder = params.get("results_folder", None)
        
        if self.results_folder is not None:
            utils.create_path(self.results_folder)

        if self.load_model_path:
            self.load_model(self.load_model_path)

    def partial_fit(self, train_data, cv_data=None):
        
        for app_name, data in train_data.items():
            if( self.verbose != 0):
                print("Preparing Dataset for %s" % app_name)

            appliance_model = self.appliances[app_name]

            timewindow = appliance_model.get("timewindow", self.default_appliance['timewindow'])
            timestep = appliance_model["timestep"]
            overlap = appliance_model.get("overlap", self.default_appliance['overlap'])
            on_treshold = appliance_model.get("on_treshold", self.default_appliance['on_treshold'])
            mains_std = appliance_model.get("mains_std", None)
            mains_mean = appliance_model.get("mains_mean", None)

            if mains_mean is None:
                X_train, mains_mean, mains_std = generate_main_timeseries(data["mains"], timewindow, timestep, overlap)
                appliance_model["mains_mean"] = mains_mean
                appliance_model["mains_std"] = mains_std
            else:
                X_train = generate_main_timeseries(data["mains"], timewindow, timestep, overlap, mains_mean, mains_std)[0]  
            
            X_train = X_train.reshape((X_train.shape[0], X_train.shape[1]))

            y_train = generate_appliance_timeseries(data["appliance"], True, timewindow, timestep, overlap, on_treshold=on_treshold)
            y_train = np.array([ -1 if x[0] == 1 else 1 for x in y_train])

            if cv_data is not None:
                X_cv = generate_main_timeseries(cv_data[app_name]["mains"], timewindow, timestep, overlap, mains_mean, mains_std)[0]
            
            X_cv = X_cv.reshape((X_cv.shape[0], X_cv.shape[1]))

            y_cv = generate_appliance_timeseries(cv_data[app_name]["appliance"], True, timewindow, timestep, overlap, on_treshold=on_treshold)
            y_cv = np.array([ -1 if x[0] == 1 else 1 for x in y_cv])

            if cv_data is not None:
                X = np.concatenate((X_train, X_cv), axis=0)
                y = np.concatenate((y_train, y_cv), axis=0)
            else:
                X = X_train
                y = y_train
            
            n_activations = sum([1 if x == 1 else 0 for x in y])

            if( self.verbose == 2):
                print("-"*5 + "Train Info" + "-"*5)
                print("Nº of examples: ", str(X.shape[0]))
                print("Nº of activations: ", str(n_activations))
                print("On Percentage: ", str(n_activations/len(y) ))
                print("Off Percentage: ", str( (len(y) - n_activations)/len(y) ))
                print("Mains Mean: ", str(mains_mean))
                print("Mains Std: ", str(mains_std))
            
            if app_name in self.model:
                if self.verbose > 0:
                    print("Starting from previous step")
                model = self.model[app_name]
            else:
                model = SVC()

            if self.verbose != 0:
                print("Training ", app_name, " in ", self.MODEL_NAME, " model\n", end="\r")
            
            #Fits the model to the training data.
            model.fit(X, y)

            #Stores the trained model.
            self.model[app_name] = model

            pred = self.model[app_name].predict(X)

            tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
            mcc = matthews_corrcoef(y, pred)

            if self.verbose == 2:
                print("Training scores")    
                print("MCC: ", mcc )
                print("True Positives: ", tp)
                print("True Negatives: ", tn)  
                print("False Negatives: ", fn)  
                print("False Positives: ", fp)  
            
            if self.results_folder is not None:
                f = open(self.results_folder + "results_" + app_name.replace(" ", "_") + ".txt", "w")
                f.write("-"*5 + "Train Info" + "-"*5+ "\n")
                f.write("Nº of examples: "+ str(X.shape[0])+ "\n")
                f.write("Nº of activations: "+ str(n_activations)+ "\n")
                f.write("On Percentage: "+ str(n_activations/len(y))+ "\n")
                f.write("Off Percentage: "+ str((len(y) - n_activations)/len(y))+ "\n")
                f.write("Mains Mean: " + str(mains_mean) + "\n")
                f.write("Mains Std: " + str(mains_std) + "\n")
                f.write("Train MCC: "+str(mcc)+ "\n")
                f.write("True Positives: "+str(tp)+ "\n")
                f.write("True Negatives: "+str(tn)+ "\n")
                f.write("False Positives: "+str(fp)+ "\n")
                f.write("False Negatives: "+str(fn)+ "\n")
                f.close()
            
    def disaggregate_chunk(self, test_mains, test_appliances):
        
        appliance_powers_dict = {}
        
        for app_name, appliance_power in test_appliances:
            if self.verbose != 0:
                print("Preparing the Test Data for %s" % app_name)
            
            appliance_model = self.appliances.get(app_name, {})
        
            timewindow = appliance_model.get("timewindow", self.default_appliance['timewindow'])
            timestep = appliance_model["timestep"]
            overlap = appliance_model.get("overlap", self.default_appliance['overlap'])
            on_treshold = appliance_model.get("on_treshold", self.default_appliance['on_treshold'])
            mains_std = appliance_model["mains_std"]
            mains_mean = appliance_model["mains_mean"]

            X_test = generate_main_timeseries(test_mains, timewindow, timestep, overlap, mains_mean, mains_std)[0]
            X_test = X_test.reshape((X_test.shape[0], X_test.shape[1]))

            if( self.verbose == 2):
                print("Nº of test examples", X_test.shape[0])

            if self.verbose != 0:
                print("Estimating power demand for '{}' in '{}'\n".format(app_name, self.MODEL_NAME))

            y_test = generate_appliance_timeseries(appliance_power, True, timewindow, timestep, overlap, on_treshold=on_treshold)
            y_test = np.array([ -1 if x[0] == 1 else 1 for x in y_test])

            n_activations = sum([1 if x == 1 else 0 for x in y_test])

            if( self.verbose != 0):
                print("Nº of examples: ", str(X_test.shape[0]))
                print("Nº of activations: ", str(n_activations))
                print("On Percentage: ", str(n_activations/len(y_test) ))
                print("Off Percentage: ", str( (len(y_test) - n_activations)/len(y_test) ))

            if self.verbose != 0:
                print("Estimating power demand for '{}' in '{}'\n".format(app_name, self.MODEL_NAME))
            
            pred = self.model[app_name].predict(X_test)
            
            tn, fn, fp, tp = confusion_matrix(y_test, pred).ravel()
            mcc = matthews_corrcoef(y_test, pred)
            if self.verbose == 2:
                print("True Positives: ", tp)
                print("True Negatives: ", tn)  
                print("False Negatives: ", fn)  
                print("False Positives: ", fp)        
                print( "MCC: ", mcc)

            if self.results_folder is not None:
                f = open(self.results_folder + "results_" + app_name.replace(" ", "_") + ".txt", "a")
                f.write("-"*5 + "Test Info" + "-"*5+ "\n")
                f.write("Nº of examples: "+ str(X_test.shape[0])+ "\n")
                f.write("Nº of activations: "+ str(n_activations)+ "\n")
                f.write("On Percentage: "+ str(n_activations/len(y_test))+ "\n")
                f.write("Off Percentage: "+ str((len(y_test) - n_activations)/len(y_test))+ "\n")
                f.write("MCC: "+str(matthews_corrcoef(y_test, pred)) + "\n")
                f.write("True Positives: "+str(tp)+ "\n")
                f.write("True Negatives: "+str(tn)+ "\n")
                f.write("False Positives: "+str(fp)+ "\n")
                f.write("False Negatives: "+str(fn)+ "\n")
                f.close()

            appliance_powers_dict[app_name] = mcc

        return appliance_powers_dict

    def save_model(self, folder_name):
        #For each appliance trained store its model
        print("save")

    def load_model(self, folder_name):
        #Get all the models trained in the given folder and load them.
        print("load")
