#train

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler, normalize, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, recall_score, precision_score, f1_score, roc_auc_score
import joblib
import json
import time

ARTIFACTS_PATH = Path("artifacts.pkl")
STATS_CLUSTER = clust_cols = ['AST', 'REB', 'STL', 'BLK', 'TOV', 'FGA', '3PA', 'FTA']
FEATURES_TRAIN = ['GP', 'MIN', 'PTS', 'FGM', 'FG%', 'FTM', 'REB', 'AST', 'STL', 'BLK', 'TOV']
TARGET = 'TARGET_5Yrs'

def kfold_scores(X, y, model):
    kf = KFold(n_splits=3, shuffle=True, random_state=50)
    cm = np.zeros((2,2))
    recalls, precs, f1s, aucs = [], [], [], []
    for tr, te in kf.split(X):
        model.fit(X[tr], y[tr])
        p = model.predict(X[te])
        proba = model.predict_proba(X[te])[:,1]
        cm += confusion_matrix(y[te], p)
        recalls.append(recall_score(y[te], p))
        precs.append(precision_score(y[te], p))
        f1s.append(f1_score(y[te], p))
        aucs.append(roc_auc_score(y[te], proba))
    return {
        "cm": (cm/3).tolist(),
        "recall": float(np.mean(recalls)),
        "precision": float(np.mean(precs)),
        "f1": float(np.mean(f1s)),
        "auc": float(np.mean(aucs)),
    }

def main(csv_path="nba_logreg.csv"):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=STATS_CLUSTER + FEATURES_TRAIN + [TARGET]).copy()

    # ===== Clustering (profil de jeu) =====
    Xc = df[STATS_CLUSTER].values
    minmax = MinMaxScaler().fit(Xc)
    Xc_scaled = minmax.transform(Xc)
    Xc_profile = normalize(Xc_scaled, norm="l1")

    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(Xc_profile)
    df["cluster"] = clusters

    # ===== Modèles par cluster =====
    models = {}                  # {cluster_id: fitted LogisticRegression}
    scalers = {}                 # {cluster_id: fitted StandardScaler}
    metrics = {}                 # {cluster_id: kfold metrics}
    for c in np.unique(clusters):
        sub = df[df["cluster"] == c]
        X = sub[FEATURES_TRAIN].values
        y = sub[TARGET].astype(int).values

        sc = StandardScaler().fit(X)
        Xz = sc.transform(X)

        clf = LogisticRegression(max_iter=10000)
        metrics[c] = kfold_scores(Xz, y, clf)
        # on refit sur tout le cluster pour la prod
        clf.fit(Xz, y)

        models[c] = clf
        scalers[c] = sc

    meta = {
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "version": "rookie-longevity-v1",
    "features_train": FEATURES_TRAIN,
    "stats_cluster": STATS_CLUSTER,
    "target": TARGET,
    "clusters_count": int(kmeans.n_clusters),
    "class_labels": [0, 1],
    # conversion explicite des clés et valeurs
    "cluster_sizes": {int(c): int((df["cluster"] == c).sum()) for c in np.unique(clusters)},
    "cv_metrics": {int(k): v for k, v in metrics.items()},
    }
    artifacts = {
        "minmax": minmax,
        "kmeans": kmeans,
        "scalers": scalers,
        "models": models,
        "meta": meta
    }

    joblib.dump(artifacts, ARTIFACTS_PATH)
    print(f"✔️  Saved artifacts to {ARTIFACTS_PATH}")
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    main()