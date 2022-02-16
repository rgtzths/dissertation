
import pandas as pd

import sys
sys.path.insert(1, "../../")
sys.path.insert(1, "../../../utils")
from sklearn.metrics import mean_squared_error
import numpy as np

def rmse(app_gt, app_pred):
    return mean_squared_error(app_gt,app_pred)**(.5)

def clean_dataframe(raw_data, smothing, window):
    """
    Parameters
    ----------
    filename : str
        File used for comparison.
    timestep : int
        Amount of seconds between samples.
    """

    if(smothing == "sma"):
        raw_data["value"] = raw_data["value"].rolling(window=window).mean()

    elif(smothing == "ema"):
        raw_data["value"] = raw_data["value"].ewm(span=window,adjust=False).mean()
    
    elif(smothing == "median"):
        raw_data["value"] = raw_data["value"].rolling(window=window).median()

    return raw_data

def gaussian_noise(noise_df, percentage, mean, std):
    random_indices = np.random.randint(0, noise_df.size, int(noise_df.size*percentage))
    noise = np.random.normal(mean, std, noise_df.size)
    if std >= 9:
        print(std, min(noise), max(noise))
    noise_df["value"][random_indices] += noise[random_indices]

    return noise_df

if __name__ == "__main__":
    input_file = "./original.csv"
    clean_methods = ["ema", "sma", "median"]
    stds = [1, 1.5, 3, 9, 27]
    windows = [3, 5, 9]

    begining = pd.to_datetime('2014-01-01')
    end = pd.to_datetime('2014-01-02')

    timestep = 6

    df = pd.read_csv(input_file, header=None, names=["time", "value"])
    df.index = pd.to_datetime(df["time"])
    df = df.drop("time", 1)
    begining_index = df.index.get_loc(begining, method="nearest")
    end_index = df.index.get_loc(end, method="nearest")
    df = df[begining_index:end_index]

    cleaned_df = df.copy(deep=True)
    noisy_df = df.copy(deep=True)

    results = {}

    for i in range(0,1):
        print("-"*5,"Epoch", i, "-"*5)
        for std in stds:

            noisy_df = df.copy(deep=True)
            noisy_df = gaussian_noise(noisy_df, 0.2, 0, std)

            for window in windows:            
                for c in clean_methods:

                    n_df = noisy_df.copy(deep=True)
                    cleaned_df = df.copy(deep=True)

                    cleaned_df = clean_dataframe(cleaned_df, c, window)                

                    # Clean the noisy df.
                    after_noise = clean_dataframe(n_df, c, window)

                    
                    #print("RMSE with std", std ,"and window", window,":", rmse(cleaned_df.dropna()["value"], after_noise.dropna()["value"]))

                    if c == "sma":
                        error = rmse(cleaned_df.dropna()["value"], after_noise["value"][window-1:])
                        #print(c, "RMSE with window", window ,":", error)
                    elif c == "median":
                        error = rmse(cleaned_df.dropna()["value"], after_noise["value"][window-1:])
                        #print(c, "RMSE with window", window ,":", error)
                    else:
                        error = rmse(cleaned_df["value"], after_noise["value"])
                        #print(c, "RMSE with window", window ,":", error)

                    if std in results:
                        if window in results[std]:
                            if c in results[std][window]:
                                results[std][window][c].append(error)
                            else:
                                results[std][window][c] = [error]
                        else:
                            results[std][window] = { c : [error]}
                    else:
                        results[std] = { window : { c : [error]}}
        
    for std in results:
        print("-"*5,"Using std:",std, "-"*5)
        for window in results[std]:
            print("-"*2,"Using window:", window, "-"*2)
            for c in results[std][window]:
                print("Average error in", c, ":", sum(results[std][window][c])/len(results[std][window][c]) )