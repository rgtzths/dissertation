import pandas as pd
import numpy as np


def clean_data(df, timestep, interpolate):

    #Definition of variables to be used during conversion
    columns = list(df.columns.values)
    offset = pd.Timedelta(int(timestep/2), unit="s")
    step = pd.Timedelta(timestep, unit="s")
    
    data = []
    index = []

    #Current time contains the fixed time that is incremented
    #with the step.
    current_time = df.index[0]
    
    if int(current_time.strftime("%S")) % timestep != 0:
        current_time += pd.Timedelta( timestep - (int(current_time.strftime("%S")) % timestep ), unit="s")

    #Current index represents the index of the dataframe.  
    current_index = 0
    
    #Makes sure that we stop at the end of the df.
    while current_index < len(df):
        feature_vector = []
        #Gets the time that corresponds to the current index
        index_time = df.index[current_index]

        #If the index time is between +/- offset of the objective time.
        #The value in that index is used for the feature vector.
        if index_time  >= current_time - offset and index_time <= current_time + offset:
            for i, c in enumerate(columns):
                if not np.isnan(df[c][df.index[current_index]]):
                    if df[c][df.index[current_index]] > 0:
                        feature_vector.append(df[c][df.index[current_index]])
                    else:
                        feature_vector.append(0)
                else:
                    feature_vector.append(data[-1][i])
            current_index += 1
            current_time += step

        #If the index time is behind the current time more than the offset
        elif current_time > index_time:
            current_index += 1
            
        #The last condition indicates that the current time is behind the index time for more than the offset.
        #In this case the average between the previous feature and the feature in the index time is used for the feature vector.
        else:
            for i, c in enumerate(columns):
                if interpolate == "average":
                    if not np.isnan(df[c][df.index[current_index]]):
                        if len(data) == 0:
                           feature_vector.append(df[c][df.index[current_index-1]] + df[c][df.index[current_index]] /2)
                        elif (df[c][df.index[current_index]] +  data[-1][i]) /2 > 0: 
                            feature_vector.append((df[c][df.index[current_index]] +  data[-1][i]) /2)
                        else:
                            feature_vector.append(0)
                    else:
                        feature_vector.append(data[-1][i])
                else:
                    feature_vector = data[-1]
            current_time += step
    
        if len(feature_vector) == len(columns):
            data.append(feature_vector)
            index.append(current_time - step)
    
    df = pd.DataFrame( data=np.array(data), index=index, columns=columns)
    df.index.name = "timestamp"
    return df