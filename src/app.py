from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd

app = FastAPI(title="Heart Disease Prediction API", version="1.0.0")

try:
    scaler = joblib.load("scaler.pkl")
    model = joblib.load("model.pkl")
except Exception as e:
    scaler, model = None, None

class PatientPayload(BaseModel):
    age: float = Field(..., example=63.0)
    sex: float = Field(..., example=1.0)
    cp: float = Field(..., example=3.0)
    trestbps: float = Field(..., example=145.0)
    chol: float = Field(..., example=233.0)
    fbs: float = Field(..., example=1.0)
    restecg: float = Field(..., example=0.0)
    thalach: float = Field(..., example=150.0)
    exang: float = Field(..., example=0.0)
    oldpeak: float = Field(..., example=2.3)
    slope: float = Field(..., example=0.0)
    ca: float = Field(..., example=0.0)
    thal: float = Field(..., example=1.0)

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict(patient: PatientPayload):
    if not model or not scaler:
        raise HTTPException(status_code=500, detail="Model or scaler artifacts missing.")

    feature_names = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
    input_df = pd.DataFrame([[
        patient.age, patient.sex, patient.cp, patient.trestbps, patient.chol,
        patient.fbs, patient.restecg, patient.thalach, patient.exang,
        patient.oldpeak, patient.slope, patient.ca, patient.thal
    ]], columns=feature_names)

    scaled = scaler.transform(input_df)
    pred = int(model.predict(scaled)[0])
    prob = float(np.max(model.predict_proba(scaled)))

    return {
        "prediction": pred,
        "diagnosis": "High Risk of Heart Disease" if pred == 1 else "Normal",
        "confidence": round(prob, 4)
    }
