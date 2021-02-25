import pandas as pd
import numpy as np

def generate_main_timeseries(dfs, is_test, timeframe, overlap, timestep):
    columns = list(dfs[0].columns.values)

    one_second = pd.Timedelta(1, unit="s")
    step = pd.Timedelta(timestep, unit="s")
    
    data = []
    aprox = 0
    arred = 0
    behind = 0
    for df in dfs:
        current_time = df.index[0].round("s", ambiguous=False)
        current_index = 0
        
        if is_test:
            objective_step = step
            objective_time = current_time + step
        else:
            objective_step = pd.Timedelta(timeframe*60, unit="s") - pd.Timedelta(timeframe*60*overlap, unit="s")
            objective_time = current_time + objective_step*2
            overlap_index =  int(timeframe*60/timestep - timeframe*60*overlap*len(columns)/timestep)

        past_feature_vector = []

        while current_index < len(df):

            feature_vector = []
            if len(past_feature_vector) != 0 and is_test:
                feature_vector = past_feature_vector[1:]
            elif len(past_feature_vector) == 0 and is_test:
                feature_vector = [0 for i in range(0, int(timeframe*60*len(columns)/timestep -1))]
            elif len(past_feature_vector) != 0:
                feature_vector = past_feature_vector[overlap_index:]

            while current_time != objective_time and current_index < len(df):
                index_time = df.index[current_index].round("s", ambiguous=False)
                if index_time == current_time or index_time - one_second == current_time or index_time + one_second == current_time:
                    for c in columns:
                        feature_vector.append(df[c[0]][c[1]][df.index[current_index]])
                    current_index += 1
                    current_time += step
                    aprox += 1
                elif current_time > index_time:
                    if  current_index < len(df) -1:
                        next_index = df.index[current_index+1].round("s", ambiguous=False)
                        if next_index == current_time or next_index - one_second == current_time:
                            for c in columns:
                                feature_vector.append(df[c[0]][c[1]][df.index[current_index+1]])
                            current_time += step
                            behind += 1
                    current_index += 2
                else:
                    for c in columns:
                        feature_vector.append((df[c[0]][c[1]][df.index[current_index]] + feature_vector[-len(columns)])/2)
                    current_time += step
                    arred += 1

            if len(feature_vector) == int(timeframe*60*len(columns)/timestep):
                objective_time += objective_step
                past_feature_vector = feature_vector
                data.append(feature_vector)
    
    return np.array(data)

def generate_appliance_timeseries(dfs, timeframe, overlap, timestep, column):
    columns = list(dfs[0].columns.values)

    one_second = pd.Timedelta(1, unit="s")
    step = pd.Timedelta(timestep, unit="s")

    aprox = 0
    arred = 0
    behind = 0
    data = []

    for df in dfs:

        current_time = df.index[0].round("s", ambiguous=False)
        current_index = 0

        objective_step = pd.Timedelta(timeframe*60, unit="s") - pd.Timedelta(timeframe*60*overlap, unit="s")
        objective_time = current_time + objective_step*2

        while current_index < len(df):
            current_value = 0
            while current_time != objective_time and current_index < len(df):
                index_time = df.index[current_index].round("s", ambiguous=False)

                if index_time == current_time or index_time - one_second == current_time or index_time + one_second == current_time:
                    current_value = df[column[0]][column[1]][df.index[current_index]]
                    current_index += 1
                    current_time += step
                    aprox += 1
                elif current_time > index_time:
                    if  current_index < len(df) -1:
                        next_index = df.index[current_index+1].round("s", ambiguous=False)
                        if next_index == current_time or next_index - one_second == current_time:
                            current_value = df[column[0]][column[1]][df.index[current_index]]
                            current_time += step
                            behind += 1
                    current_index += 2
                else:
                    current_value = (df[column[0]][column[1]][df.index[current_index]] + data[-1] )/2 
                    current_time += step
                    arred += 1

            if current_time == objective_time:
                data.append(current_value)
            objective_time += objective_step

    return np.array(data)