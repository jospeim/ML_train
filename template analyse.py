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
from sklearn.pipeline import Pipeline
import shap

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
plt.title("Corrélation entre variables numériques et " + nom_cible)
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
plt.title("Corrélation (Cramér’s V) avec la variable cible " + nom_cible, fontsize=14, pad=12)
plt.xlabel("Cramér’s V")
plt.ylabel("")
plt.xlim(0, 1)
plt.tight_layout()
plt.show()

df_encoded = pd.get_dummies(df, columns=cat_features, drop_first=True)
X = df_encoded.drop(columns=[nom_cible])
y = df_encoded[nom_cible]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Pipeline : normalisation + modèle
pipe_lr = Pipeline([
    ("scaler", StandardScaler()),
    ("log_reg", LogisticRegression(max_iter=10000))
])

# Entraînement
pipe_lr.fit(X_train, y_train)

explainer = shap.Explainer(pipe_lr.named_steps["log_reg"], X_train, feature_names=X.columns)
shap_values = explainer(X_test)

# Afficher un résumé global des importances
shap.summary_plot(shap_values, X_test, plot_type="bar")


rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))
print("AUC random forest:", roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]))

importances_rf = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
plt.figure(figsize=(8,6))
importances_rf.head(10).plot(kind="barh", color="darkorange")
plt.title("Top 10 features importantes (Random Forest)")
plt.show()

score_rf = cross_val_score(rf, X, y, cv=5, scoring="roc_auc").mean()
print(f"AUC moyenne Random Forest : {score_rf:.3f}")
