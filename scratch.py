import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor


df = pd.read_csv("C:/Users/joach/OneDrive/Documents/entrainement/archive/student-mat.csv", sep = ",")
print(df.columns)

age = df[["age"]]
print(np.unique(age))

plt.hist(age)

for y in np.unique(age):
    print(y, len(df[df["age"] == y]))
plt.show()

X, y = df.iloc[:, :-1], df.iloc[:, -1]
cat_cols = X.select_dtypes(include=["object"]).columns   # colonnes catégorielles
num_cols = X.select_dtypes(exclude=["object"]).columns   # colonnes numériques

# Préprocessing : OneHot pour les catégories, StandardScaler pour les numériques
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

# Pipeline complet avec un RandomForest
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor())
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Entraînement
model.fit(X_train, y_train)

# Évaluation
print("Score R² :", model.score(X_test, y_test))