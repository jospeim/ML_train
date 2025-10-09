import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest

df = pd.read_csv("C:/Users/joach/OneDrive/Documents/entrainement/telecom/WA_Fn-UseC_-Telco-Customer-Churn.csv")
# print(df.isnull().sum())
print(df.columns)
print(df.describe())
df.loc[df["Churn"] == "No", "Churn"] = 0
df.loc[df["Churn"] == "Yes", "Churn"] = 1
df["Churn"] = df["Churn"].astype(int)

corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.show()

# --- 1️⃣ Détection des colonnes catégorielles ---
cat_features = df.select_dtypes(include="object").columns

# --- 2️⃣ Fonction de calcul du Cramér’s V ---
def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k - 1)*(r - 1)) / (n - 1))
    rcorr = r - ((r - 1)**2) / (n - 1)
    kcorr = k - ((k - 1)**2) / (n - 1)
    return np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))

# --- 3️⃣ Calcul du Cramér’s V pour chaque variable catégorielle ---
cramers_results = pd.DataFrame({
    "Feature": cat_features,
    "CramersV": [cramers_v(df[col], df["Churn"]) for col in cat_features]
}).sort_values("CramersV", ascending=False).reset_index(drop=True)

print("📊 Corrélation (Cramér’s V) entre variables catégorielles et Churn :")
print(cramers_results)

# --- 4️⃣ Heatmap visuelle ---
plt.figure(figsize=(8, 0.4 * len(cramers_results)))
sns.heatmap(
    cramers_results.set_index("Feature").T,
    annot=True, cmap="coolwarm", cbar=False, fmt=".2f"
)
plt.yticks([])
plt.show()
