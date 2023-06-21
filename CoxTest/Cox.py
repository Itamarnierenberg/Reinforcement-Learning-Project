import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

data = pd.read_csv("./dataset/lung.csv", index_col = 0)
print(f'[INFO] Samples From the Data:')
print(data.head())
print(f'[INFO] Data Shape = {data.shape}')
data = data[['time', 'status', 'age', 'sex', 'ph.ecog', 'ph.karno','pat.karno', 'meal.cal', 'wt.loss']]
data["status"] = data["status"] - 1
data["sex"] = data["sex"] - 1
print(f'[INFO] Samples From the Data:')
print(data.head())
print(f'[INFO] Data Types = {data.dtypes}')
print(f'[INFO] Missing Data =\n{data.isnull().sum()}')

data["ph.karno"].fillna(data["ph.karno"].mean(), inplace = True)
data["pat.karno"].fillna(data["pat.karno"].mean(), inplace = True)
data["meal.cal"].fillna(data["meal.cal"].mean(), inplace = True)
data["wt.loss"].fillna(data["wt.loss"].mean(), inplace = True)
data.dropna(inplace=True)
data["ph.ecog"] = data["ph.ecog"].astype("int64")

T = data["time"]
E = data["status"]
#plt.hist(T, bins = 50)
#plt.show()

kmf = KaplanMeierFitter()
kmf.fit(durations = T, event_observed = E)
kmf.plot_survival_function()
kmf.survival_function_.plot(plt.gca())
plt.title('Survival function')