from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from src.serving.model_loader import load_model
from src.monitoring.drift import detect_drift, drift_detected

app = FastAPI()
model = load_model()
baseline = pd.read_parquet("data/credit_train.parquet").drop(columns=["default"])

class Applicant(BaseModel):
    age: int
    income: float
    credit_lines: int
    delinquencies: int
    utilization: float
    months_active: int
    region: str

@app.post("/predict")
def predict(applicant: Applicant):
    X = pd.DataFrame([applicant.dict()])
    proba = float(model.predict_proba(X)[0,1])
    return {"probability_default": proba}

@app.post("/drift")
def check_drift(new_data: list[dict]):
    current = pd.DataFrame(new_data)
    report = detect_drift(baseline, current)
    return {
        "drift_detected": drift_detected(report),
        "report": report
    }