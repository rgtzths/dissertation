import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=RuntimeWarning)

from os.path import join, isfile
from os import listdir

from keras.models import Sequential, load_model
from keras.layers import Dense, GRU
from keras.wrappers.scikit_learn import KerasClassifier

from scipy.stats import randint
from sklearn.metrics import matthews_corrcoef, confusion_matrix, make_scorer
from sklearn.model_selection import RandomizedSearchCV

import sys
sys.path.insert(1, "../feature_extractors")
from generate_timeseries import generate_main_timeseries, generate_appliance_timeseries
from matthews_correlation import matthews_correlation

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
        #Defines the number of nodes in the GRU, 
        #Normaly varies between 0.5 and 2 time the input size.
        self.n_nodes = params.get('n_nodes', {})
        #Number of epochs that the models run
        self.epochs = params.get('epochs', {})
        #Number of examples presented per batch
        self.batch_size = params.get('batch_size', {})
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
                if( self.verbose != 0):
                    print("Preparing Dataset for %s" % app_name)
                
                #Get the timewindow and timestep
                timewindow = self.timewindow.get(app_name, 5)
                timestep = self.timestep.get(app_name, 6)

                X_train = generate_main_timeseries(train_mains, False, timewindow, timestep)  
        
                y_train = generate_appliance_timeseries(appliance_power, True, timewindow, timestep, self.column)

                #Divides the training set into training and cross validations
                X_cv = X_train[int(len(X_train)*(1-self.cv)):]
                y_cv = y_train[int(len(y_train)*(1-self.cv)):]

                X_train = X_train[0:int(len(X_train)*(1-self.cv))]
                y_train = y_train[0:int(len(y_train)*(1-self.cv))]

                if self.verbose != 0:
                    print("Training ", app_name, " in ", self.MODEL_NAME, " model\n", end="\r")
                
                #Checks if the model already exists and if it doesn't creates a new one.
                if app_name in self.model:
                    model = self.model[app_name]
                else:
                    model = self.create_model(self.n_nodes.get(app_name, X_train.shape[1]), (X_train.shape[1], X_train.shape[2]))

                #Fits the model to the training data.
                model.fit(X_train, y_train, epochs=self.epochs.get(app_name, 200), batch_size=self.batch_size.get(app_name, 500), validation_data=(X_cv, y_cv), verbose=self.verbose, shuffle=False)
                #Stores the trained model.
                self.model[app_name] = model
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

            timewindow = self.timewindow.get(app_name, 5)
            timestep = self.timestep.get(app_name, 6)

            X_test = generate_main_timeseries(test_mains, False, timewindow, timestep) 

            y_test = generate_appliance_timeseries(appliance_power, True, timewindow, timestep, self.column)
            
            if self.verbose != 0:
                print("Estimating power demand for '{}' in '{}'\n".format(app_name, self.MODEL_NAME))
            
            pred = self.model[app_name].predict(X_test)
            pred = [ 0 if p < 0.5 else 1 for p in pred ]
        
            y_test = y_test.reshape(len(y_test),)
            
            tn, fn, fp, tp = confusion_matrix(y_test, pred).ravel()

            if self.verbose == 2:
                print("True Positives: ", tp)
                print("True Negatives: ", tn)  
                print("False Negatives: ", fn)  
                print("False Positives: ", fp)        
                print( "MCC: ", matthews_corrcoef(y_test, pred))

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
            self.model[app.split(".")[0]] = load_model(join(folder_name, app))

    def create_model(self, n_nodes, input_shape):
        #Creates a specific model.
        model = Sequential()
        model.add(GRU(n_nodes, input_shape=input_shape))
        model.add(Dense(units=1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[matthews_correlation])

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