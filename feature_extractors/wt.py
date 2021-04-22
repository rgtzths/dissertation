import numpy as np
import pywt
from collections import Counter
from scipy import stats
import math

from seasonality import get_seasonality

def calculate_entropy(list_values):
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1]/len(list_values) for elem in counter_values]
    entropy=stats.entropy(probabilities)
    return [entropy]
 
def calculate_statistics(list_values):
    n5 = np.nanpercentile(list_values, 5)
    n25 = np.nanpercentile(list_values, 25)
    n75 = np.nanpercentile(list_values, 75)
    n95 = np.nanpercentile(list_values, 95)

    median = np.nanpercentile(list_values, 50)
    mean = np.nanmean(list_values)
    
    std = np.nanstd(list_values)
    var = np.nanvar(list_values)
    
    rms = np.nanmean(np.sqrt(list_values**2))

    mode, count = stats.mode(list_values)
    skew = stats.skew(list_values)
    min_value = np.min(list_values)
    max_value = np.max(list_values)
    
    return [n5, n25, n75, n95, median, mean, std, var, rms, mode[0], skew, min_value, max_value]
 
def calculate_crossings(list_values):
    mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
    no_mean_crossings = len(mean_crossing_indices)
    return [no_mean_crossings]
 
#def get_discrete_features(data, waveletname):
#    X = []
#
#    for signal_no in range(0, len(data)):
#        features = []
#        for signal_comp in range(0,data.shape[2]):
#            signal = data[signal_no, :, signal_comp]
#            list_coeff = pywt.wavedec(signal, waveletname)
#            for coeff in list_coeff:
#                entropy = calculate_entropy(coeff)
#                crossings = calculate_crossings(coeff)
#                statistics = calculate_statistics(coeff)
#                features += [entropy] + crossings + statistics
#        X.append(features)
#    return np.array(X)
#
#def get_continuous_features(data, waveletname):
#
#    X = np.ndarray(shape=(data.shape[0], data.shape[1]-1, data.shape[1]-1, data.shape[2]))
#
#    for i in range(0, data.shape[0]):
#        for j in range(0, data.shape[2]):
#            signal = data[i, :, j]
#            coeff, freq = pywt.cwt(signal, range(1, data.shape[1]), waveletname, 1)
#            coeff_ = coeff[:,:data.shape[1]-1]
#            X[i, :, :, j] = coeff_ 
#    
#    return X

def get_discrete_features(dfs, waveletname, timestep, dwt_timewindow, dwt_overlap, examples_timewindow, examples_overlap):
    
    n_columns = len(dfs[0].columns.values)

    step = int((dwt_timewindow-dwt_overlap)/timestep) 

    X = []
    
    window_size = int(examples_timewindow / (dwt_timewindow - dwt_overlap))

    overlap_index = int((examples_timewindow - examples_overlap) / (dwt_timewindow- dwt_overlap))
    for df in dfs:
        current_index = 0

        window_vectors = []
        for i in range(0, math.ceil( examples_overlap / (dwt_timewindow - dwt_overlap))):
            window_vectors.append(list(np.zeros(15*n_columns + 9)))

        if dwt_overlap != 0:
            values = np.zeros((math.ceil(dwt_overlap/timestep), n_columns))
        else:
            values = np.zeros((0, n_columns))

        while current_index + step <= len(df):

            values = np.append(values, df.loc[df.index[current_index:current_index + step].values].values, axis=0)

            if values.shape[0] * values.shape[1] != int(dwt_timewindow*n_columns/timestep):
                raise Exception("Invalid length of values ", values.shape[0] * values.shape[1], int(dwt_timewindow*n_columns/timestep))

            feature_vector = []
            
            for signal_comp in range(0,values.shape[1]):
                list_coeff = pywt.wavedec(values[:, signal_comp], waveletname)
                for coeff in list_coeff:
                    entropy = calculate_entropy(coeff)
                    crossings = calculate_crossings(coeff)
                    statistics = calculate_statistics(coeff)
                    feature_vector += entropy + crossings + statistics

            feature_vector += get_seasonality(df.index[current_index + step -1])
            
            window_vectors.append(feature_vector)
            if len(window_vectors) == window_size:

                X.append(window_vectors)

                window_vectors = window_vectors[overlap_index:]
            
            values = values[step:,:]
            current_index += step

    return np.array(X)

def get_continuous_features(data, waveletname, timewindow, timestep, overlap):

    n_columns = len(dfs[0].columns.values)

    overlap_index = int((timewindow-overlap)*n_columns/timestep)
    step = int((timewindow-overlap)/timestep)
    
    n_examples = 0

    for df in dfs:
        n_examples += int(df.shape[0]*timestep/(timewindow-overlap))

    X = np.ndarray(shape=(n_examples, int(timewindow/timestep), int(timewindow/timestep), n_columns))
   
    i = 0
    
    for df in dfs:
        current_index = 0
        previous_values = []

        while current_index + step < len(df):

            if len(previous_values) == 0:
                values = np.zeros(overlap_index)
            else:
                values = previous_values

            values = np.append(values, df.loc[df.index[current_index:current_index + step ].values].values)

            previous_values = values[overlap_index:]

            values = values.reshape(1, timewindow/timestep, n_columns)

            for j in range(0, values.shape[2]):
                signal = values[0, :, j]
                coeff, freq = pywt.cwt(signal, range(1, int(timewindow/overlap), waveletname, 1))
                coeff_ = coeff[:, : int(timewindow/overlap)-1]
                X[i, :, :, j] = coeff_ 
            
            i += 1
        
            current_index += overlap

    return X