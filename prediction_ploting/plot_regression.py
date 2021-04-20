
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import warnings
from nilmtk.measurement import LEVEL_NAMES
from gru_dwt import GRU_DWT

### Valores Editáveis #####
app_name = "heatpump"

beginning = pd.to_datetime('2020-12-02T18:45')
end = pd.to_datetime('2020-12-02T19:15')

model_path = "../regression_models/"

######

column_mapping = {
    "power" : ("power", "apparent"),
    "vrms" : ("voltage", "")
}

app_file = "../../datasets/avEiro_classification/house_1/"+ app_name +".csv"

df = pd.read_csv(app_file, sep=',', header=[0,1], index_col=0)
df.columns = pd.MultiIndex.from_tuples(df.columns)
df.index = pd.to_datetime(df.index)
df.columns = pd.MultiIndex.from_tuples([column_mapping["power"]])
df.columns.set_names(LEVEL_NAMES, inplace=True)

beginning_index = df.index.get_loc(beginning, method="nearest")

end_index = df.index.get_loc(end, method="nearest")

app_file = "../../datasets/avEiro_classification/house_1/mains.csv"

df2 = pd.read_csv(app_file, sep=',', header=[0,1], index_col=0)
df2.columns = pd.MultiIndex.from_tuples(df2.columns)
df2.index = pd.to_datetime(df2.index)
df2.columns = pd.MultiIndex.from_tuples([column_mapping["power"], column_mapping["vrms"]])
df2.columns.set_names(LEVEL_NAMES, inplace=True)

beginning_index_2 = df2.index.get_loc(beginning, method="nearest")

end_index_2 = df2.index.get_loc(end, method="nearest")

model = GRU_DWT({"load_model_folder": model_path})
predictions = model.disaggregate_chunk(df["power"]["apparent"][beginning_index:end_index])

print(predictions)

plt.plot(df.index[beginning_index:end_index], df["power"]["apparent"][beginning_index:end_index], label="Heatpump")
plt.plot(df2.index[beginning_index_2:end_index_2], df2["power"]["apparent"][beginning_index_2:end_index_2], label="Aggregated")
plt.plot(df2.index[beginning_index_2:end_index_2], predictions, label="Predicted_Heatpump")

plt.xlabel("Time")
plt.ylabel("Power")
plt.title("Aggregated Readings, HeatPump readings and predicted readings")
plt.legend()

plt.show()