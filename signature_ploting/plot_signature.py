
import pandas as pd
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
warnings.filterwarnings('ignore',category=RuntimeWarning)

### Valores Editáveis #####
app_name = "heatpump"

begining = pd.to_datetime('2020-12-01T012:30')
end = pd.to_datetime('2020-12-01T13:30')
######


app_file = "../../datasets/avEiro_dataset_v2/house_1/"+ app_name +"/power.csv"

df = pd.read_csv(app_file)
df["time"] = pd.to_datetime(df["time"], unit='ns')
df.index = df["time"]
df.index = df.index.round("T", ambiguous=False)

begining_index = 0

while(df.index[begining_index] != begining):
    begining_index += 1

end_index = begining_index

while(df.index[end_index] != end):
    end_index += 1


plt.plot(df["time"][begining_index:end_index], df["value"][begining_index:end_index])
plt.xlabel("Time")
plt.ylabel("Power")
plt.title(app_name)
plt.show()