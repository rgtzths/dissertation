import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=RuntimeWarning)
from os.path import join, isfile
from os import listdir
from nilmtk.disaggregate import Disaggregator
from keras.models import Sequential, load_model
from keras.layers import Dense, LSTM
import pandas as pd
from generate_timeseries import generate_main_timeseries, generate_appliance_timeseries

class LSTM_RNN(Disaggregator):
    def __init__(self, params):
        self.model = {}
        self.MODEL_NAME = 'LSTM RNN'
        self.cv = params.get('cv', 0.18)
        self.timeframe = params.get('timeframe', 5)
        self.timestep = params.get('timestep', 2)
        self.overlap = params.get('overlap', 0.5)
        self.column = params.get('predicted_column', ("power", "apparent"))
        self.load_model_path = params.get('load_model_folder',None)
        self.input_size = params.get('input_size', 0)
        self.epochs = params.get('epochs', 300)
        self.verbose = params.get('verbose', 0)

        if self.load_model_path:
            self.load_model(self.load_model_path)

    def partial_fit(self, train_main, train_appliances, **load_kwargs):
        n_features = len(train_main[0].columns.values)

        n_past_examples = int(self.timeframe*60/self.timestep)

        print("Preparing the Training Data: X")

        X_train = generate_main_timeseries(train_main, False, self.timeframe, self.overlap, self.timestep)

        X_train = X_train.reshape(X_train.shape[0], n_past_examples, n_features)
        
        X_cv = X_train[int(len(X_train)*(1-self.cv)):]
        
        X_train = X_train[0:int(len(X_train)*(1-self.cv))]

        for app_name, power in train_appliances:
            print("Preparing the Training Data: Y")
            y_train = generate_appliance_timeseries(power, self.timeframe, self.overlap, self.timestep, self.column)
            
            y_cv = y_train[int(len(y_train)*(1-self.cv)):]
            y_train = y_train[0:int(len(y_train)*(1-self.cv))]
           
            print("Training ", app_name, " in ", self.MODEL_NAME, " model\n", end="\r")
            
            if app_name in self.model:
                model = self.model[app_name]
            else:
                model = Sequential()
                model.add(LSTM(X_train.shape[1] + X_train.shape[1]*self.input_size, input_shape=(n_past_examples, n_features)))
                model.add(Dense(1))
                model.compile(optimizer='adam', loss='mean_squared_error', metrics=["RootMeanSquaredError"])

            model.fit(X_train, y_train, epochs=self.epochs, batch_size=1000, validation_data=(X_cv, y_cv), verbose=self.verbose, shuffle=False)

            self.model[app_name] = model
            
    def disaggregate_chunk(self, test_mains):
        test_predictions_list = []

        n_past_examples = int(self.timeframe*60/self.timestep)
        
        n_features = len(test_mains[0].columns.values)

        print("Preparing the Test Data")
        X_test = generate_main_timeseries(test_mains, True, self.timeframe, self.overlap, self.timestep)
        
        X_test = X_test.reshape(X_test.shape[0], n_past_examples, n_features)
        
        appliance_powers_dict = {}

        for i, app_name in enumerate(self.model):
        
            print("Estimating power demand for '{}' in '{}'\n".format(app_name, self.MODEL_NAME))
            pred = self.model[app_name].predict(X_test)
            pred = [p[0] for p in pred]

            column = pd.Series(
                    pred, index=test_mains[0].index, name=i)
            appliance_powers_dict[app_name] = column
        
        test_predictions_list.append(pd.DataFrame(appliance_powers_dict, dtype='float32'))

        return test_predictions_list

    def save_model(self, folder_name):
        for app in self.model:
            self.model[app].save(join(folder_name, app + ".h5"))

    def load_model(self, folder_name):
        app_models = [f for f in listdir(folder_name) if isfile(join(folder_name, f))]
        for app in app_models:
            self.model[app.split(".")[0]] = load_model(join(folder_name, app))
