
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=RuntimeWarning)

### Valores Editáveis #####
app_name = "heatpump"

beginning = pd.to_datetime('2020-12-02T18:45')
end = pd.to_datetime('2020-12-02T19:15')
######


app_file = "../../datasets/avEiro/house_1/"+ app_name +"/power.csv"

df = pd.read_csv(app_file)

df.index = pd.to_datetime(df["time"])

beginning_index = df.index.get_loc(beginning, method="nearest")

end_index = df.index.get_loc(end, method="nearest")


app_file = "../../datasets/avEiro/house_1/mains/power.csv"

df2 = pd.read_csv(app_file)

df2.index = pd.to_datetime(df2["time"])

beginning_index_2 = df2.index.get_loc(beginning, method="nearest")

end_index_2 = df2.index.get_loc(end, method="nearest")

plt.plot(df.index[beginning_index:end_index], df["value"][beginning_index:end_index], label="Heatpump")
plt.plot(df2.index[beginning_index_2:end_index_2], df2["value"][beginning_index_2:end_index_2], label="Aggregated")

plt.xlabel("Time")
plt.ylabel("Power")
plt.title("Aggregated Readings and HeatPump readings")
plt.legend()

plt.show()