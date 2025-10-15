# ===============================
# 0️⃣ Importations
# ===============================
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ===============================
# 1️⃣ DataFrame d'entraînement et de test
# ===============================
name_test = "C:/Users/joach/OneDrive/Documents/entrainement/test_hmp/TF1_test_set.xlsx"
name = "C:/Users/joach/OneDrive/Documents/entrainement/test_hmp/TF1_train_set.csv"

df_train = pd.read_csv(name, sep = ";").iloc[:500, :]
df_test = pd.read_excel(name_test).iloc[:500, :]
df_test = df_test.dropna()
df_tot = df_train.merge(df_test, on="visitor_id")
df_tot = df_tot.rename(columns={'visitor_id': 'Années'})
df_train = df_tot.iloc[:,:-1]
df_test = pd.concat([df_tot.iloc[:, :1], df_tot.iloc[:, -1:]], axis=1)
df_train["cible"] = df_test.iloc[:,1]

plt.figure(figsize=(12,8))
plt.hist(df_train["cible"], bins=np.arange(0.5, 8.5, 1), rwidth=0.8)
plt.title("Distribution de la cible")
plt.savefig("image/distribution_classification")
plt.show()