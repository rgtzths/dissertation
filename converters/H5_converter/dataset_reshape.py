
import pandas as pd

def reshape(input_path):
    df = pd.read_csv(input_path)
    df.drop("name", axis=1, inplace=True)
    df.to_csv(input_path, index=False, header=False)

filename="../avEiro/house_1/channel_2.csv"

reshape(filename)