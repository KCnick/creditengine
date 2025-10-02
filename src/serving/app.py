from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from model_loader import load_model

app = FastAPI()
model = load_model()

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
