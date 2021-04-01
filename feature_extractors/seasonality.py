import datetime

def get_seasonality(dt):
    month = dt.month
    feature_vector = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    if month <= 3:
        i = 0
        value = month % 4
    elif month <=  6:
        i = 1
        value = (month+1) % 4 

    elif month <= 9:
        i = 2
        value = (month+2) % 4
    else:
        i = 3
        value = (month+3) % 4
    feature_vector[i] = value

    if dt.weekday() < 5 :
        feature_vector[4] = dt.weekday() + 1 
    else:
        feature_vector[5] = dt.weekday() - 4
    
    if dt.hour > 6 and dt.hour < 13:
        feature_vector[6] = 1
    elif dt.hour > 13 and dt.hour < 20:
        feature_vector[7] = 1
    else:
        feature_vector[8] = 1
    


    return feature_vector