# --- 1️⃣ Importations et chargement des données ---
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as ss
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report

predicted = "classification"
name = "C:/Users/joach/OneDrive/Documents/entrainement/telecom/WA_Fn-UseC_-Telco-Customer-Churn.csv"
nettoyage = True
nom_cible = "Churn"

def clean_and_convert_numeric(df):
    # 1️⃣ Supprimer les lignes avec NaN
    df = df.dropna().copy()

    # 2️⃣ Parcourir les colonnes de type object
    for col in df.select_dtypes(include="object").columns:
        # Nettoyage basique des valeurs
        df[col] = df[col].astype(str).str.strip()  # retire les espaces
        df[col] = df[col].replace(["", "-", "N/A", "NaN", "nan"], np.nan)
        df[col] = df[col].str.replace(",", ".", regex=False)  # virgules -> points

        # 3️⃣ Conversion en numérique (si possible)
        converted = pd.to_numeric(df[col], errors="coerce")

        # Si plus de 95 % des valeurs sont numériques → on garde la conversion
        if converted.notna().mean() > 0.95:
            df[col] = converted
            # Si toutes les valeurs sont des entiers (pas de décimales) → convertir en int
            if (df[col].dropna() % 1 == 0).all():
                df[col] = df[col].astype("Int64")
            else:
                df[col] = df[col].astype(float)

    return df.dropna()


col_to_keep = []
col_num = []

df = pd.read_csv(name)
df[nom_cible] = df[nom_cible].map({"Yes": 1, "No": 0}).astype(int)
df = df.drop(columns=["customerID"])
if nettoyage:
    df = clean_and_convert_numeric(df)
print("nombre de lignes nulles", df.isnull().sum())
print("type des colonnes", df.dtypes)

# Distribution de la cible
sns.countplot(x=nom_cible, data=df)
plt.title("Répartition du Churn")
plt.show()

# Aperçu des variables numériques
num_features = df.select_dtypes(include=[np.number]).columns
print(df[num_features].describe())
print("features numériques : ",num_features)

corr = df[num_features].corr()
plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Corrélation entre variables numériques")
plt.show()


cat_features = df.select_dtypes(include="object").columns
print("features catégoriques : ",cat_features)

def cramers_v(x, y):
    confusion_matrix = pd.crosstab(x, y)
    chi2 = ss.chi2_contingency(confusion_matrix)[0]
    n = confusion_matrix.sum().sum()
    phi2 = chi2/n
    r, k = confusion_matrix.shape
    phi2corr = max(0, phi2 - ((k-1)*(r-1))/(n-1))
    rcorr = r - ((r-1)**2)/(n-1)
    kcorr = k - ((k-1)**2)/(n-1)
    return np.sqrt(phi2corr / min((kcorr-1), (rcorr-1)))

cramers_results = pd.DataFrame({
    "Feature": cat_features,
    "CramersV": [cramers_v(df[col], df["Churn"]) for col in cat_features]
}).sort_values("CramersV", ascending=False)

plt.figure(figsize=(8, 0.5 * len(cramers_results)))
sns.barplot(
    data=cramers_results,
    y="Feature",
    x="CramersV",
    palette="coolwarm"
)
plt.title("Corrélation (Cramér’s V) avec la variable cible Churn", fontsize=14, pad=12)
plt.xlabel("Cramér’s V")
plt.ylabel("")
plt.xlim(0, 1)
plt.tight_layout()
plt.show()

df_encoded = pd.get_dummies(df, columns=cat_features, drop_first=True)

# print("ok", df_encoded.columns)
# print("re", df_encoded.info())
X = df_encoded.drop(columns=["Churn"])
y = df_encoded["Churn"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
# print(num_features, type(num_features))
X_train[num_features.drop("Churn")] = scaler.fit_transform(X_train[num_features.drop("Churn")])
X_test[num_features.drop("Churn")] = scaler.transform(X_test[num_features.drop("Churn")])
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)
print(classification_report(y_test, y_pred))
print("AUC :", roc_auc_score(y_test, log_reg.predict_proba(X_test)[:,1]))

# Importance des features
importances_lr = pd.Series(log_reg.coef_[0], index=X_train.columns).sort_values(ascending=False)
plt.figure(figsize=(8,6))
importances_lr.head(10).plot(kind="barh", color="teal")
plt.title("Top 10 features importantes (Logistic Regression)")
plt.show()

rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))
print("AUC :", roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]))

importances_rf = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
plt.figure(figsize=(8,6))
importances_rf.head(10).plot(kind="barh", color="darkorange")
plt.title("Top 10 features importantes (Random Forest)")
plt.show()

score_lr = cross_val_score(log_reg, X, y, cv=5, scoring="roc_auc").mean()
score_rf = cross_val_score(rf, X, y, cv=5, scoring="roc_auc").mean()

print(f"AUC moyenne Logistic Regression : {score_lr:.3f}")
print(f"AUC moyenne Random Forest : {score_rf:.3f}")
