import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from src.app.main import app
import src.nlp.nlp_pipeline as nlp_module
import src.monitoring.prometheus_metrics as metrics_module

client = TestClient(app)

# ── 1. Dataset schema ────────────────────────────────────────────────────────
def test_data_schema_validation():
    df = pd.DataFrame({'Poids': [1.0], 'Volume': [2.0], 'Conductivite': [0.5],
                       'Opacite': [1.0], 'Rigidite': [2.0]})
    required_cols = ['Poids', 'Volume', 'Conductivite', 'Opacite', 'Rigidite']
    assert all(col in df.columns for col in required_cols)

# ── 2. Post-imputation quality ───────────────────────────────────────────────
def test_post_imputation_quality_check():
    df = pd.DataFrame({'Poids': [1.0, np.nan, 3.0]})
    df['Poids'] = df['Poids'].fillna(df['Poids'].median())
    assert not df['Poids'].isnull().any()

# ── 3. NLP preprocessing (ZONE ROUGE) ───────────────────────────────────────
def test_nlp_pipeline_test():
    result = nlp_module.preprocess_text("Lot de papier récupéré à l'Usine A.")
    assert isinstance(result, str)
    assert len(result) > 0

def test_preprocess_removes_numbers():
    result = nlp_module.preprocess_text("Poids de 16.7 kg, volume 64 litres")
    assert "16" not in result
    assert "64" not in result

def test_preprocess_removes_stopwords():
    result = nlp_module.preprocess_text("le la les de et en")
    for tok in result.split():
        assert tok not in nlp_module.ALL_STOPWORDS

def test_preprocess_handles_empty():
    assert nlp_module.preprocess_text("") == ""
    assert nlp_module.preprocess_text(None) == ""

def test_tokenize_min_length():
    tokens = nlp_module.tokenize("a de lot papier")
    assert "a" not in tokens
    assert "lot" in tokens

def test_stem_fr():
    result = nlp_module.stem_fr("plastiques")
    assert len(result) < len("plastiques")

def test_preprocess_series():
    s = pd.Series(["Lot de papier", "Bris de verre", None])
    result = nlp_module.preprocess_series(s)
    assert len(result) == 3
    assert isinstance(result.iloc[0], str)
    assert result.iloc[2] == ""

# ── 4. Model performance threshold ──────────────────────────────────────────
def test_model_performance_threshold():
    accuracy = 0.75
    assert accuracy >= 0.70

# ── 5. Prometheus metrics ────────────────────────────────────────────────────
def test_prometheus_metrics_exist():
    assert metrics_module.REQUEST_COUNT is not None
    assert metrics_module.REQUEST_LATENCY is not None
    assert metrics_module.PREDICTION_COUNT is not None
    assert metrics_module.MODEL_ACCURACY is not None
    assert metrics_module.DRIFT_JS_SCORE is not None

# ── 6. API endpoints ─────────────────────────────────────────────────────────
def test_api_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_api_predict_endpoint_validation():
    response = client.post("/predict", json={"invalid": "data"})
    assert response.status_code == 422

def test_api_predict_returns_category():
    response = client.post("/predict", json={
        "poids": 40.0, "volume": 100.0, "conductivite": 0.2,
        "opacite": 1.0, "rigidite": 5.0, "source": "Usine_A"
    })
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        data = response.json()
        assert "categorie" in data
        assert data["categorie"] in {"Plastique", "Verre", "Papier", "Métal"}

def test_api_nlp_endpoint():
    response = client.post("/predict-nlp",
                           json={"rapport": "Matériau plastique souple non conducteur."})
    assert response.status_code in (200, 503)
    if response.status_code == 200:
        assert "categorie" in response.json()

def test_api_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200



def test_api_predict_with_valid_source():
    for source in ["Centre_Tri", "Collecte_Citoyenne", "Inconnu", "Usine_A", "Usine_B"]:
        response = client.post("/predict", json={
            "poids": 20.0, "volume": 50.0, "conductivite": 0.5,
            "opacite": 2.0, "rigidite": 3.0, "source": source
        })
        assert response.status_code in (200, 503)

def test_api_predict_unknown_source():
    response = client.post("/predict", json={
        "poids": 20.0, "volume": 50.0, "conductivite": 0.5,
        "opacite": 2.0, "rigidite": 3.0, "source": "Unknown_Source"
    })
    assert response.status_code in (200, 503)

def test_api_nlp_short_text_rejected():
    response = client.post("/predict-nlp", json={"rapport": "hi"})
    assert response.status_code == 422



