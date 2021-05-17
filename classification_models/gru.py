import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=RuntimeWarning)

from os.path import join, isfile
from os import listdir
import json

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, GRU
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from scipy.stats import randint
from sklearn.metrics import matthews_corrcoef, confusion_matrix, make_scorer
from sklearn.model_selection import RandomizedSearchCV

import numpy as np

import sys
sys.path.insert(1, "../feature_extractors")
from generate_timeseries import generate_main_timeseries, generate_appliance_timeseries
from matthews_correlation import matthews_correlation

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

class GRU_RNN():
    def __init__(self, params):
        #Variable that will store the models trained for each appliance.
        self.model = {}
        #Name used to identify the Model Name.
        self.MODEL_NAME = params.get('model_name', 'GRU')
        #Percentage of values used as cross validation data from the training data.
        self.cv = params.get('cv', 0.16)
        #Column used to obtain the classification values (y). 
        self.column = params.get('predicted_column', ("power", "apparent"))
        #If this variable is not None than the class loads the appliance models present in the folder.
        self.load_model_path = params.get('load_model_folder',None)
        #Dictates the ammount of information to be presented during training and regression.
        self.verbose = params.get('verbose', 0)

        #Defines the window of time of each feature vector (in mins)
        self.timewindow = params.get('timewindow', {})
        #Defines the step bertween each reading (in seconds)
        self.timestep = params.get('timestep', {})
        self.overlap = params.get('overlap', {})

        self.n_nodes = params.get('n_nodes', {})
        #Number of epochs that the models run
        self.epochs = params.get('epochs', {})
        #Number of examples presented per batch
        self.batch_size = params.get('batch_size', {})

        self.training_results_path = params.get("training_results_path", None)
        self.checkpoint_file = params.get("checkpoint_file", None)
        self.results_file = params.get("results_path", None)


        #Decides if the model runs randomsearch
        self.randomsearch = params.get('randomsearch', False)
        #In case of randomsearch = True this variable contains the information
        #to run the randomsearch (hyperparameter range definition).
        self.randomsearch_params = params.get('randomsearch_params', None)

        if self.load_model_path:
            self.load_model(self.load_model_path)

    def partial_fit(self, train_mains, train_appliances):
        #Checks the need to do random search.       
        if not self.randomsearch:
            #For each appliance to be classified
            for app_name, appliance_power in train_appliances:
                if app_name not in self.model:
                    if( self.verbose != 0):
                        print("Preparing Dataset for %s" % app_name)

                    X_train = generate_main_timeseries(train_mains, self.timewindow, self.timestep, self.overlap)  

                    y_train = generate_appliance_timeseries(appliance_power, True, self.timewindow, self.timestep, self.column, self.overlap)

                    if( self.verbose != 0):
                        print("Nº de Positivos ", sum([ np.where(p == max(p))[0][0]  for p in y_train]))
                        print("Nº de Negativos ", y_train.shape[0]-sum([ np.where(p == max(p))[0][0]  for p in y_train ]))

                    if self.verbose != 0:
                        print("Training ", app_name, " in ", self.MODEL_NAME, " model\n", end="\r")
                    
                    #Checks if the model already exists and if it doesn't creates a new one.
                    if app_name in self.model:
                        model = self.model[app_name]
                    else:
                        model = self.create_model(self.n_nodes, (X_train.shape[1], X_train.shape[2]))

                    checkpoint = ModelCheckpoint(self.checkpoint_file, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
                    #Fits the model to the training data.
                    history = model.fit( X_train, 
                            y_train, 
                            epochs=self.epochs, 
                            batch_size=self.batch_size, 
                            verbose=self.verbose, 
                            shuffle=True,
                            callbacks=[checkpoint],
                            validation_split=self.cv,
                            )
                    history = json.dumps(history.history)

                    if self.training_results_path is not None:
                        f = open(self.training_results_path + "history_"+app_name+"_"+self.MODEL_NAME+".json", "w")
                        f.write(history)
                        f.close()

                    model.load_weights(self.checkpoint_file)
                    
                    #Stores the trained model.
                    self.model[app_name] = model

                    if self.results_file is not None:
                        f = open(self.results_file, "w")
                        f.write("Nº de Positivos para treino: " + str(sum([ np.where(p == max(p))[0][0]  for p in y_train])) + "\n")
                        f.write("Nº de Negativos para treino: " + str(y_train.shape[0]-sum([ np.where(p == max(p))[0][0]  for p in y_train ])) + "\n")
                        f.close()
                else:
                    print("Using Loaded Model")
        else:
            if self.verbose != 0:
                print("Executing RandomSearch")
            results = self.random_search(train_mains, train_appliances)

            if self.verbose == 2:
                for app in results:
                    print("\nResults for appliance: ", app)
                    print("\t Score Obtained: ", str(results[app][0]))
                    print("\t Best Parameters: ", str(results[app][1]))
                    print("\t Time Window: ", str(results[app][2]))
                    print("\t Time Step: ", str(results[app][3]))
                    print("\n")
            
    def disaggregate_chunk(self, test_mains, test_appliances):
        
        appliance_powers_dict = {}
        
        for app_name, appliance_power in test_appliances:
            if self.verbose != 0:
                print("Preparing the Test Data for %s" % app_name)

            X_test = generate_main_timeseries(test_mains, self.timewindow, self.timestep, self.overlap) 

            y_test = generate_appliance_timeseries(appliance_power, True, self.timewindow, self.timestep, self.column, self.overlap)
            
            if( self.verbose != 0):
                print("Nº de Positivos ", sum([ np.where(p == max(p))[0][0]  for p in y_test ]))
                print("Nº de Negativos ", y_test.shape[0]-sum([ np.where(p == max(p))[0][0]  for p in y_test ]))

            if self.verbose != 0:
                print("Estimating power demand for '{}' in '{}'\n".format(app_name, self.MODEL_NAME))
            
            pred = self.model[app_name].predict(X_test)
            pred = [ np.where(p == max(p))[0][0]  for p in pred ]
        
            y_test = [ np.where(p == max(p))[0][0]  for p in y_test ]
            
            tn, fn, fp, tp = confusion_matrix(y_test, pred).ravel()

            if self.verbose == 2:
                print("True Positives: ", tp)
                print("True Negatives: ", tn)  
                print("False Negatives: ", fn)  
                print("False Positives: ", fp)        
                print( "MCC: ", matthews_corrcoef(y_test, pred))

            if self.results_file is not None:
                f = open(self.results_file, "a")
                f.write("Nº de Positivos para teste: " + str(sum(y_test)) + "\n")
                f.write("Nº de Negativos para teste: " + str(len(y_test)- sum(y_test)) + "\n")
                f.write("MCC: "+str(matthews_corrcoef(y_test, pred)))
                f.write("True Positives: "+str(tp)+ "\n")
                f.write("True Negatives: "+str(tn)+ "\n")
                f.write("False Positives: "+str(fp)+ "\n")
                f.write("False Negatives: "+str(fn)+ "\n")
                f.close()

            appliance_powers_dict[app_name] = matthews_corrcoef(y_test, pred)

        return appliance_powers_dict

    def save_model(self, folder_name):
        #For each appliance trained store its model
        for app in self.model:
            self.model[app].save(join(folder_name, app + "_" + self.MODEL_NAME + ".h5"))

    def load_model(self, folder_name):
        #Get all the models trained in the given folder and load them.
        app_models = [f for f in listdir(folder_name) if isfile(join(folder_name, f))]
        for app in app_models:
            self.model[app.split(".")[0].split("_")[2]] = load_model(join(folder_name, app), custom_objects={"matthews_correlation":matthews_correlation})

    def create_model(self, n_nodes, input_shape):
        #Creates a specific model.
        model = Sequential()
        model.add(GRU(n_nodes, input_shape=input_shape, activation="relu"))
        model.add(Dense(2, activation='softmax'))
        model.compile(optimizer=Adam(0.00001), loss='categorical_crossentropy', metrics=["accuracy", matthews_correlation])

        return model

    def random_search(self, train_mains, train_appliances):

        #Stores the parameters for the best model for each appliance
        test_results = {}

        for timewindow in self.randomsearch_params['timewindow']:
            for timestep in self.randomsearch_params['timestep']:
                
                #Obtains de X training data acording to the timewindow and timestep
                X_train = generate_main_timeseries(train_mains, False, timewindow, timestep)  

                #Defines the input shape according to the timewindow and timestep.
                self.randomsearch_params['model']['input_shape'] = [(X_train.shape[1], X_train.shape[2])]

                self.randomsearch_params['model']['n_nodes'] = randint(int(X_train.shape[1] *0.5), int(X_train.shape[1]*2))
                
                for app_name, appliance_power in train_appliances:
                    #Generate the appliance timeseries acording to the timewindow and timestep
                    y_train = generate_appliance_timeseries(appliance_power, True, timewindow, timestep, self.column)

                    model = KerasClassifier(build_fn = self.create_model, verbose=0)
                    
                    randomsearch = RandomizedSearchCV(
                        estimator=model,
                        param_distributions=self.randomsearch_params['model'], 
                        cv=self.randomsearch_params['cv'], 
                        n_iter=self.randomsearch_params['n_iter'],
                        n_jobs=self.randomsearch_params['n_jobs'], 
                        scoring=make_scorer(matthews_corrcoef),
                        verbose=self.verbose,
                        refit = False
                    )

                    fitted_model = randomsearch.fit(X_train, y_train)

                    #Store the best result, if it is actualu the best and the X and y used for the final training.
                    if app_name not in test_results :
                        test_results[app_name] = (fitted_model.best_score_, fitted_model.best_params_, timewindow, timestep)

                    elif test_results[app_name][0] < fitted_model.best_score_:
                        test_results[app_name] = (fitted_model.best_score_, fitted_model.best_params_, timewindow, timestep)

        #Train the final model with the best hyperparameters for each appliance
        for app_name in test_results:
            
            X_train = generate_main_timeseries(train_mains, False, test_results[app_name][2], test_results[app_name][3]) 
                        
            X_cv = X_train[int(len(X_train)*(1-self.cv)):]
            X_train = X_train[0:int(len(X_train)*(1-self.cv))]

            y_train = generate_appliance_timeseries(appliance_power, True, test_results[app_name][2], test_results[app_name][3], self.column)

            y_cv = y_train[int(len(y_train)*(1-self.cv)):]
            y_train = y_train[0:int(len(y_train)*(1-self.cv))]

            print("Training ", app_name, " in ", self.MODEL_NAME, " model\n", end="\r")
            model = self.create_model( test_results[app_name][1]['n_nodes'], test_results[app_name][1]['input_shape'])

            model.fit(X_train, y_train, epochs=test_results[app_name][1]['epochs'], batch_size=test_results[app_name][1]['batch_size'], validation_data=(X_cv, y_cv), verbose=self.verbose, shuffle=True)

            self.model[app_name] = model

            self.timewindow[app_name] = test_results[app_name][2]

            self.timestep[app_name] = test_results[app_name][3]
            
        return test_results