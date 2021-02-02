import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=RuntimeWarning)

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import matthews_corrcoef, make_scorer
import keras.backend as K
import pandas as pd
import numpy as np

def matthews_correlation(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())

class LSTM_RNN(Disaggregator):
    def __init__(self, timeframe, timestep, overlap, cv=0.2 ):
        self.MODEL_NAME = 'LSTM RNN'
        self.cv = cv
        self.timeframe = timeframe
        self.timestep = timestep
        self.overlap = overlap
        self.model = {}

    def partial_fit(self, train_main, train_appliances, **load_kwargs):
        n_features = len(train_main[0].columns.values)
        n_past_examples = int(self.timeframe*60/self.timestep)
        print("Preparing the Training Data")

        X_train = self.generate_timeseries(train_main[0], True)

        X_train = X_train.reshape(X_train.shape[0], n_past_examples, n_features)
        
        X_cv = X_train[int(len(X_train)*(1-self.cv)):]
        
        X_train = X_train[0:int(len(X_train)*(1-self.cv))]

        for app_name, power in train_appliances:
            print("Preparing the Test Data")
            y_train = self.generate_timeseries(power[0], False)
            
            y_cv = y_train[int(len(y_train)*(1-self.cv)):]
            y_train = y_train[0:int(len(y_train)*(1-self.cv))]
           
            print("Training ", app_name, " in ", self.MODEL_NAME, " model\n", end="\r")
            
            model = Sequential()
            model.add(LSTM(256, input_shape=(n_past_examples, n_features)))
            model.add(Dense(units=1, activation='sigmoid'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[matthews_correlation])
            model.fit(X_train, y_train, epochs=1, batch_size=72, validation_data=(X_cv, y_cv), verbose=2, shuffle=False)

            self.model[app_name] = model
            
    def disaggregate_chunk(self, test_mains):
        test_predictions_list = []
        n_past_examples = int(self.timeframe*60/self.timestep)

        X_test = self.generate_timeseries(test_mains[0], True)
        X_test = X_test.reshape(X_test.shape[0], n_past_examples, n_features)

        appliance_powers_dict = {}

        for i, app_name in enumerate(self.model):
        
            print("Estimating power demand for '{}' in '{}'\n".format(app_name, self.MODEL_NAME))
            pred = self.model[app_name].predict(X_test)

            column = pd.Series(
                    pred, index=test_mains[0].index, name=i)
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

    def generate_timeseries(self, df, is_mains):
        one_second = pd.Timedelta(1, unit="s")
        timestep = pd.Timedelta(self.timestep, unit="s")
        overlap_index =  self.timeframe*30 - int(self.timeframe*30*self.overlap) -1
        objective_step = pd.Timedelta(self.timeframe*60, unit="s") - pd.Timedelta(self.timeframe*60*self.overlap, unit="s")

        current_time = df.index[0].round("s", ambiguous=False)
        current_index = 0
        objective_time = current_time + pd.Timedelta(self.timeframe*60, unit="s")

        past_feature_vector = []
        aprox = 0
        arred = 0
        behind = 0
        data = []

        columns = list(df.columns.values)

        while current_index < len(df):
            feature_vector = []
            if len(past_feature_vector) != 0:
                feature_vector = past_feature_vector[ overlap_index : -1]

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
            if current_index < len(df):
                objective_time += objective_step
                past_feature_vector = feature_vector
                if is_mains:
                    data.append(feature_vector)
                else:
                    added = False
                    for i in feature_vector:
                        if i > 0:
                            data.append(1)
                            added = True
                            break
                    if not added:
                        data.append(0)
        
        print("")
        print("Aprox Values: ", aprox)
        print("Arred values: ", arred)
        print("Behind values:", behind)
        print("")
        return np.array(data)
