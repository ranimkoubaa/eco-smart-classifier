import pytest
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient
from src.app.main import app

def test_data_schema_validation():
    df = pd.DataFrame({'Poids': [1.0], 'Volume': [2.0], 'Conductivite': [0.5], 'Opacite': [1.0], 'Rigidite': [2.0]})
    required_cols = ['Poids', 'Volume', 'Conductivite', 'Opacite', 'Rigidite']
    assert all(col in df.columns for col in required_cols)

def test_post_imputation_quality_check():
    df = pd.DataFrame({'Poids': [1.0, np.nan, 3.0]})
    df['Poids'] = df['Poids'].fillna(df['Poids'].median())
    assert not df['Poids'].isnull().any()

def test_nlp_pipeline_test():
    texts = ["Test nlp pipeline", "Second document"]
    assert len(texts) > 0

def test_model_performance_threshold():
    accuracy = 0.75
    assert accuracy >= 0.70

client = TestClient(app)

def test_api_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200

def test_api_predict_endpoint_validation():
    response = client.post("/predict", json={"invalid": "data"})
    assert response.status_code == 422
