from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import pandas as pd

app = FastAPI(title="Eco-Smart Classifier API")

class PredictRequest(BaseModel):
    Poids: float
    Volume: float
    Conductivite: float
    Opacite: float
    Rigidite: float
    Rapport_Collecte: str

class PredictResponse(BaseModel):
    prediction: str

model = None

@app.on_event("startup")
def load_model():
    global model
    model_path = os.path.join("models", "best_model.pkl")
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
        except Exception:
            pass

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        df = pd.DataFrame([request.dict()])
        pred = model.predict(df)[0]
        return PredictResponse(prediction=str(pred))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
