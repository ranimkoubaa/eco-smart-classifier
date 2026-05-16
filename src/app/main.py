"""
Eco-Smart Classifier — FastAPI REST API  (Personne 3)
Endpoints: GET /health  POST /predict  POST /predict-nlp  GET /metrics

Input features match exactly what Personne 1 outputs in train.csv:
  Numeric (standardized): Poids, Volume, Conductivite, Opacite, Rigidite, Prix_Revente
  OHE Source: Source_Centre_Tri, Source_Collecte_Citoyenne, Source_Inconnu,
              Source_Usine_A, Source_Usine_B
  Source_encoded (label encoded int)
"""

import logging
import joblib
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field

from src.monitoring.prometheus_metrics import (
    PREDICTION_CONFIDENCE, PREDICTION_COUNT,
    REQUEST_COUNT, REQUEST_LATENCY,
)

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")

# ── Known sources (must match Personne 1's LabelEncoder order) ───────────────
SOURCES = ["Centre_Tri", "Collecte_Citoyenne", "Inconnu", "Usine_A", "Usine_B"]
SOURCE_LABEL = {s: i for i, s in enumerate(SOURCES)}

store: dict = {}


def _load(path: str, key: str) -> None:
    if Path(path).exists():
        store[key] = joblib.load(path)
        log.info("Loaded %s", key)
    else:
        log.warning("Not found: %s", path)


@asynccontextmanager
async def lifespan(app: FastAPI):
    with open("params.yaml") as f:
        store["params"] = yaml.safe_load(f)
    _load("models/classifier.pkl",    "classifier")   # Personne 2 RandomForest
    _load("models/regressor.pkl",     "regressor")    # Personne 2 RandomForest
    _load("models/nlp_classifier.pkl","nlp")          # Personne 3 TF-IDF + SVM
    yield


app = FastAPI(
    title="Eco-Smart Classifier API",
    description="Waste category classification and resale price estimation.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])


# ── Schemas ──────────────────────────────────────────────────────────────────
class NumericInput(BaseModel):
    poids:         float = Field(..., description="Weight kg")
    volume:        float = Field(..., description="Volume L")
    conductivite:  float = Field(..., description="Conductivity")
    opacite:       float = Field(..., description="Opacity")
    rigidite:      float = Field(..., ge=1, le=10, description="Rigidity 1-10")
    source: Optional[str] = Field("Inconnu",
                                   description="Centre_Tri | Collecte_Citoyenne | Inconnu | Usine_A | Usine_B")


class NLPInput(BaseModel):
    rapport: str = Field(..., min_length=5,
                         description="French collection report text")


class PredictionOut(BaseModel):
    categorie:   str
    confidence:  Optional[float] = None
    prix_estime: Optional[float] = None
    model_used:  str


class NLPOut(BaseModel):
    categorie:  str
    model_used: str = "nlp_tfidf_svm"


# ── Helper: build feature row matching train.csv column order ────────────────
def _build_feature_row(inp: NumericInput) -> pd.DataFrame:
    source = inp.source if inp.source in SOURCE_LABEL else "Inconnu"
    row = {
        "Poids":                     inp.poids,
        "Volume":                    inp.volume,
        "Conductivite":              inp.conductivite,
        "Opacite":                   inp.opacite,
        "Rigidite":                  inp.rigidite,
        "Source_Centre_Tri":         int(source == "Centre_Tri"),
        "Source_Collecte_Citoyenne": int(source == "Collecte_Citoyenne"),
        "Source_Inconnu":            int(source == "Inconnu"),
        "Source_Usine_A":            int(source == "Usine_A"),
        "Source_Usine_B":            int(source == "Usine_B"),
        "Source_encoded":            SOURCE_LABEL[source],
    }
    return pd.DataFrame([row])


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health", tags=["meta"])
def health():
    return {"status": "ok", "models_loaded": list(store.keys())}

@app.get("/debug-features")
def debug_features():
    if "classifier" not in store:
        return {"error": "no classifier"}
    clf = store["classifier"]
    n_features = clf.n_features_in_ if hasattr(clf, "n_features_in_") else "unknown"
    feature_names = clf.feature_names_in_.tolist() if hasattr(clf, "feature_names_in_") else "unknown"
    return {"n_features": n_features, "feature_names": feature_names}

@app.get("/metrics", tags=["meta"])
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/predict", response_model=PredictionOut, tags=["inference"])
def predict(inp: NumericInput):
    t0 = time.time()
    try:
        if "classifier" not in store:
            raise HTTPException(503, "Classifier not loaded")

        X   = _build_feature_row(inp)
        clf = store["classifier"]
        cat = clf.predict(X)[0]

        conf = None
        if hasattr(clf, "predict_proba"):
            conf = float(np.max(clf.predict_proba(X)[0]))
            PREDICTION_CONFIDENCE.labels(model="numeric").observe(conf)

        prix = None
        if "regressor" in store:
            # Regressor uses same features minus Prix_Revente
            X_reg = X.drop(columns=["Prix_Revente"], errors="ignore")
            prix  = round(float(store["regressor"].predict(X_reg)[0]), 2)

        REQUEST_COUNT.labels(endpoint="/predict", status_code="200").inc()
        PREDICTION_COUNT.labels(category=cat, model="numeric").inc()
        REQUEST_LATENCY.labels(endpoint="/predict").observe(time.time() - t0)

        return PredictionOut(categorie=cat, confidence=conf,
                             prix_estime=prix, model_used="RandomForest")

    except HTTPException:
        raise
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/predict", status_code="500").inc()
        log.exception("Predict error")
        raise HTTPException(500, str(e))


@app.post("/predict-nlp", response_model=NLPOut, tags=["inference"])
def predict_nlp(inp: NLPInput):
    t0 = time.time()
    try:
        if "nlp" not in store:
            raise HTTPException(503, "NLP model not loaded")

        from src.nlp.nlp_pipeline import preprocess_text
        cat = store["nlp"].predict([preprocess_text(inp.rapport)])[0]

        REQUEST_COUNT.labels(endpoint="/predict-nlp", status_code="200").inc()
        PREDICTION_COUNT.labels(category=cat, model="nlp").inc()
        REQUEST_LATENCY.labels(endpoint="/predict-nlp").observe(time.time() - t0)

        return NLPOut(categorie=cat)

    except HTTPException:
        raise
    except Exception as e:
        REQUEST_COUNT.labels(endpoint="/predict-nlp", status_code="500").inc()
        log.exception("NLP predict error")
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.app.main:app", host="0.0.0.0", port=8000, reload=False)

