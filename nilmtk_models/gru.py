import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=RuntimeWarning)

from nilmtk.disaggregate import Disaggregator
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras.backend as K
import pandas as pd
import numpy as np

class GRU_RNN(Disaggregator):
    def __init__(self, timeframe, timestep, predicted_column, cv=0.2 ):
        self.MODEL_NAME = 'GRU RNN'
        self.cv = cv
        self.timeframe = timeframe
        self.timestep = timestep
        self.overlap = overlap
        self.model = {}
        self.column = predicted_column

    def partial_fit(self, train_main, train_appliances, **load_kwargs):
        n_features = len(train_main[0].columns.values)
        n_past_examples = int(self.timeframe*60/self.timestep)

        print("Preparing the Training Data")

        X_train = self.generate_main_timeseries(train_main[0], False)
        X_train = X_train.reshape(X_train.shape[0], n_past_examples, n_features)
        
        X_cv = X_train[int(len(X_train)*(1-self.cv)):]
        
        X_train = X_train[0:int(len(X_train)*(1-self.cv))]

        for app_name, power in train_appliances:
            print("Preparing the Test Data")
            y_train = self.generate_appliance_timeseries(power[0])
            
            y_cv = y_train[int(len(y_train)*(1-self.cv)):]
            y_train = y_train[0:int(len(y_train)*(1-self.cv))]
           
            print("Training ", app_name, " in ", self.MODEL_NAME, " model\n", end="\r")
            
            model = Sequential()
            model.add(GRU(X_train.shape[1], input_shape=(n_past_examples, n_features)))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error', metrics=["RootMeanSquaredError"])
            model.fit(X_train, y_train, epochs=1, batch_size=500, validation_data=(X_cv, y_cv), verbose=2, shuffle=False)

            self.model[app_name] = model
            
    def disaggregate_chunk(self, test_mains):
        test_predictions_list = []
        n_past_examples = int(self.timeframe*60/self.timestep)
        n_features = len(test_mains[0].columns.values)

        X_test = self.generate_main_timeseries(test_mains[0], True)
        X_test = X_test.reshape(X_test.shape[0], n_past_examples, n_features)
        
        appliance_powers_dict = {}

        for i, app_name in enumerate(self.model):
        
            print("Estimating power demand for '{}' in '{}'\n".format(app_name, self.MODEL_NAME))
            pred = self.model[app_name].predict(X_test)
            pred = [p[0] for p in pred]
            column = pd.Series(
                    pred, index=test_mains[0].index, name=i)
            print(column)
            appliance_powers_dict[app_name] = column
            
        appliance_powers = pd.DataFrame(
                appliance_powers_dict, dtype='float32')
        test_predictions_list.append(appliance_powers)

        return test_predictions_list

    def save_model(self, folder_name):
        #TODO
        return

    def load_model(self, folder_name):
        #TODO
        return

    def generate_main_timeseries(self, df):
        columns = list(df.columns.values)

        one_second = pd.Timedelta(1, unit="s")
        timestep = pd.Timedelta(self.timestep, unit="s")
        current_time = df.index[0].round("s", ambiguous=False)
        current_index = 0

        objective_step = timestep
        objective_time = current_time + timestep

        past_feature_vector = []
        aprox = 0
        arred = 0
        behind = 0
        data = []

        while current_index < len(df):

            feature_vector = []
            if len(past_feature_vector) != 0:
                feature_vector = past_feature_vector[ 1:]
            elif len(past_feature_vector) == 0:
                feature_vector = [0 for i in range(0, int(self.timeframe*60*len(columns)/self.timestep -1))]

            while current_time != objective_time and current_index < len(df):
                index_time = df.index[current_index].round("s", ambiguous=False)
                if index_time == current_time or index_time - one_second == current_time or index_time + one_second == current_time:
                    for c in columns:
                        feature_vector.append(df[c[0]][c[1]][df.index[current_index]])
                    current_index += 1
                    current_time += timestep
                    aprox += 1
                elif current_time > index_time:
                    if  current_index < len(df) -1:
                        next_index = df.index[current_index+1].round("s", ambiguous=False)
                        if next_index == current_time or next_index - one_second == current_time:
                            for c in columns:
                                feature_vector.append(df[c[0]][c[1]][df.index[current_index+1]])
                            current_time += timestep
                            behind += 1
                    current_index += 2
                else:
                    for c in columns:
                        feature_vector.append((df[c[0]][c[1]][df.index[current_index]] + feature_vector[-len(columns)])/2)
                    current_time += timestep
                    arred += 1

            if len(feature_vector) == int(self.timeframe*60*len(columns)/self.timestep):
                objective_time += objective_step
                past_feature_vector = feature_vector
                data.append(feature_vector)
        
        print("")
        print("Aprox Values: ", aprox)
        print("Arred values: ", arred)
        print("Behind values:", behind)
        print("")
        return np.array(data)
    
    def generate_appliance_timeseries(self, df):
        columns = list(df.columns.values)

        one_second = pd.Timedelta(1, unit="s")
        timestep = pd.Timedelta(self.timestep, unit="s")
        current_time = df.index[0].round("s", ambiguous=False)
        current_index = 0

        objective_step = timestep
        objective_time = current_time + timestep

        aprox = 0
        arred = 0
        behind = 0
        data = []

        while current_index < len(df):

            while current_time != objective_time and current_index < len(df):
                index_time = df.index[current_index].round("s", ambiguous=False)

                if index_time == current_time or index_time - one_second == current_time or index_time + one_second == current_time:
                    data.append(df[self.column[0]][self.column[1]][df.index[current_index]])
                    current_index += 1
                    current_time += timestep
                    aprox += 1
                elif current_time > index_time:
                    if  current_index < len(df) -1:
                        next_index = df.index[current_index+1].round("s", ambiguous=False)
                        if next_index == current_time or next_index - one_second == current_time:
                            data.append(df[self.column[0]][self.column[1]][df.index[current_index]])
                            current_time += timestep
                            behind += 1
                    current_index += 2
                else:
                    data.append( (df[self.column[0]][self.column[1]][df.index[current_index]] + data[-1] )/2 )
                    current_time += timestep
                    arred += 1

            objective_time += objective_step
        
        print("")
        print("Aprox Values: ", aprox)
        print("Arred values: ", arred)
        print("Behind values:", behind)
        print("")
        return np.array(data)