import numpy as np
import pandas as pd
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

def calculate_wavelet(values, n_columns, waveletname):
    values = values.reshape((-1, n_columns))
    feature_vector = []
    
    for signal_comp in range(0,values.shape[1]):
        list_coeff = pywt.wavedec(values[:, signal_comp], waveletname)
        for coeff in list_coeff:
            entropy = calculate_entropy(coeff)
            crossings = calculate_crossings(coeff)
            statistics = calculate_statistics(coeff)
            feature_vector += entropy + crossings + statistics
    return feature_vector

def get_discrete_features(dfs, waveletname, timestep, timewindow, overlap, mains_mean=None, mains_std=None):
    X = []
    
    n_columns = len(dfs[0].columns.values)

    step = int((timewindow - overlap)*n_columns/timestep)

    window_size = int(timewindow * n_columns /timestep)

    pad = window_size - step
    
    for df in dfs:
                
        new_mains = df.values.flatten()

        new_mains = np.pad(new_mains, (pad, 0),'constant', constant_values=(0,0))
        
        new_mains = np.array([ calculate_wavelet(new_mains[i : i + window_size], n_columns, waveletname) for i in range(0, len(new_mains) - window_size + 1, step)])

        X.append(pd.DataFrame(new_mains))

    X = pd.concat(X, axis=0).values
    
    if mains_mean is None:
        mains_mean = np.mean(X, axis=0)
        mains_std = np.std(X, axis=0)

    X = (X - mains_mean) / mains_std

    return X, mains_mean, mains_std

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