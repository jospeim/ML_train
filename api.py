# api.py
from typing import Optional, Dict, Any
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, root_validator, Field

# ============
# Chargement
# ============
ARTIFACTS_PATH = "artifacts.pkl"
artifacts = joblib.load(ARTIFACTS_PATH)

MINMAX = artifacts["minmax"]
KMEANS = artifacts["kmeans"]
SCALERS = artifacts["scalers"]
MODELS = artifacts["models"]
META = artifacts["meta"]

FEATURES_TRAIN = META["features_train"]
STATS_CLUSTER  = META["stats_cluster"]

UNION_FEATURES = list(dict.fromkeys(STATS_CLUSTER + FEATURES_TRAIN))

# ---------------
# Schéma d'entrée
# ---------------
class PlayerInput(BaseModel):
    # Colonnes de STATS_CLUSTER
    AST: float
    REB: float
    STL: float
    BLK: float
    TOV: float
    FGA: float
    THREEPA: float = Field(..., alias="3PA")
    FTA: float

    # Colonnes de FEATURES_TRAIN
    GP: float
    MIN: float
    PTS: float
    FGM: float
    FG_pct: float = Field(..., alias="FG_pct")
    FTM: float
    # Duplicatas déjà déclarés: REB, AST, STL, BLK, TOV (réutilisés)

    # Mapping d'alias "dangereux" (ex: FG%) et autres variantes
    @root_validator(pre=True)
    def normalize_aliases(cls, values: Dict[str, Any]):
        # 1) Supporter la clé 'FG%' en entrée (convertie vers FG_pct)
        if "FG%" in values and "FG_pct" not in values:
            values["FG_pct"] = values["FG%"]

        # 2) Supporter la clé brute '3PA' (déjà aliasée vers _3PA), OK

        # 3) Pour être indulgent sur la casse / espaces
        normalized = {}
        for k, v in values.items():
            nk = k.strip()
            if nk.lower() == "fg%":
                nk = "FG_pct"
            normalized[nk] = v
        return normalized

    class Config:
        allow_population_by_field_name = True
        anystr_strip_whitespace = True
        extra = "ignore"  # ignorer les champs inconnus


# ---------------
# Schéma de sortie
# ---------------
class PredictionOut(BaseModel):
    cluster: int
    prob_5yrs: float
    class_5yrs: int
    details: Dict[str, Any]


app = FastAPI(
    title="NBA Rookie Longevity API test",
    description="Prédiction unitaire: probabilité qu'un rookie reste ≥5 ans en NBA (modèle par cluster).",
    version=META.get("version", "v1"),
)


@app.get("/health")
def health():
    return {"status": "ok", "model_version": META.get("version"), "clusters_count": META.get("clusters_count")}


@app.get("/meta")
def meta():
    return {
        "version": META.get("version"),
        "created_at": META.get("created_at"),
        "features_train": FEATURES_TRAIN,
        "stats_cluster": STATS_CLUSTER,
        "clusters_count": META.get("clusters_count"),
        "cluster_sizes": META.get("cluster_sizes"),
        "cv_metrics": META.get("cv_metrics"),
    }


def _to_cluster_vector(payload: PlayerInput) -> np.ndarray:
    """
    Construit le vecteur pour le clustering KMeans (ordre = STATS_CLUSTER),
    en appliquant MinMax + normalisation L1 comme à l'entraînement.
    """
    # Reconstituer les valeurs dans l'ordre
    # '3PA' a été stocké dans le modèle en tant que champ _3PA côté Pydantic
    row = {
        "AST": payload.AST,
        "REB": payload.REB,
        "STL": payload.STL,
        "BLK": payload.BLK,
        "TOV": payload.TOV,
        "FGA": payload.FGA,
        "3PA": payload.THREEPA,
        "FTA": payload.FTA,
    }
    x = np.array([[row[c] for c in STATS_CLUSTER]], dtype=float)
    x_scaled = MINMAX.transform(x)

    # Normalisation L1 manuelle
    sums = np.abs(x_scaled).sum(axis=1, keepdims=True)
    sums[sums == 0.0] = 1.0
    x_profile = x_scaled / sums
    return x_profile


def _to_model_vector(payload: PlayerInput, cluster_id: int) -> np.ndarray:
    """
    Construit le vecteur X pour le modèle du cluster (ordre = FEATURES_TRAIN),
    puis applique le StandardScaler du cluster.
    """
    # Construire un dict fusionné avec tous les champs requis
    # On prend les valeurs à partir du payload + FG_pct déjà normalisé
    as_dict = {
        "GP": payload.GP,
        "MIN": payload.MIN,
        "PTS": payload.PTS,
        "FGM": payload.FGM,
        "FG%": payload.FG_pct,   # le modèle a été entraîné avec la colonne 'FG%'
        "FTM": payload.FTM,
        "REB": payload.REB,
        "AST": payload.AST,
        "STL": payload.STL,
        "BLK": payload.BLK,
        "TOV": payload.TOV,
    }

    # Respecter l'ordre exact de FEATURES_TRAIN
    x = np.array([[as_dict[c] for c in FEATURES_TRAIN]], dtype=float)

    if cluster_id not in SCALERS or cluster_id not in MODELS:
        raise HTTPException(status_code=400, detail=f"Aucun scaler/modèle trouvé pour le cluster {cluster_id}.")

    scaler = SCALERS[cluster_id]
    xz = scaler.transform(x)
    return xz

profil = ["meneur","attaquant","défenseur"]
noms = ["hors de la ligue", "toujours dans la ligue"]

# Nouveau schéma de sortie adapté à tes textes
class PredictionOut(BaseModel):
    cluster: str
    probabilite_de_duree_5_ans: str
    prediction_a_5_ans: str


@app.post("/predict", response_model=PredictionOut)
def predict(payload: PlayerInput):
    # 1) Assigner le cluster
    xc = _to_cluster_vector(payload)
    cluster_id = int(KMEANS.predict(xc)[0])

    # 2) Construire le vecteur pour le modèle du cluster
    xz = _to_model_vector(payload, cluster_id)

    # 3) Inférence
    model = MODELS[cluster_id]
    prob = float(model.predict_proba(xz)[0, 1])
    yhat = int(prob >= 0.5)

    # 4) Réponse formatée
    return PredictionOut(
        cluster=profil[cluster_id],
        probabilite_de_duree_5_ans=f"{prob * 100:.1f}%",   # formaté à 1 décimale
        prediction_a_5_ans=noms[yhat],
    )

