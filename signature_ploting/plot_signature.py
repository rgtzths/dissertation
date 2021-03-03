
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=RuntimeWarning)

### Valores Editáveis #####
app_name = "heatpump"

beginning = pd.to_datetime('2020-10-02T01:00')
end = pd.to_datetime('2020-12-01T13:30')
######


app_file = "../../datasets/avEiro/house_1/"+ app_name +"/power.csv"

df = pd.read_csv(app_file)

df.index = pd.to_datetime(df["time"])

beginning_index = df.index.get_loc(beginning, method="nearest")

end_index = df.index.get_loc(end, method="nearest")


plt.plot(df["time"][beginning_index:end_index], df["value"][beginning_index:end_index])
plt.xlabel("Time")
plt.ylabel("Power")
plt.title(app_name)
plt.show()