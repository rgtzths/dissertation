import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import matthews_corrcoef, make_scorer
import keras.backend as K
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=RuntimeWarning)

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

class LSTM_RNN():
    def __init__(self, time_lag, n_features, cv=0.16 ):
        self.MODEL_NAME = 'LSTM RNN'
        
        model = Sequential()
        model.add(LSTM(300, input_shape=(time_lag, n_features)))
        model.add(Dense(units=1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[matthews_correlation])

        self.model = model
        self.cv = cv

    def partial_fit(self, X_train, y_train):
        X_cv = X_train[int(len(X_train)*(1-self.cv)):]
        y_cv = y_train[int(len(y_train)*(1-self.cv)):]

        X_train = X_train[0:int(len(X_train)*(1-self.cv))]
        y_train = y_train[0:int(len(y_train)*(1-self.cv))]
        
        self.model.fit(X_train, y_train, epochs=500, batch_size=500, validation_data=(X_cv, y_cv), verbose=2, shuffle=False)
            
    def disaggregate_chunk(self, X_test, y_test):
        pred = self.model.predict(X_test)
        pred = [ 0 if p < 0.5 else 1 for p in pred ]
        tp = 0
        tn = 0
        fp = 0
        fn = 0

        for i in range(0, len(pred)):
            if pred[i] == y_test[i] and y_test[i] == 1:
                tp +=1
            elif pred[i] == y_test[i]:
                tn += 1
            elif pred[i] != y_test[i] and y_test[i] == 1:
                fn += 1
            else:
                fp += 1
        
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
