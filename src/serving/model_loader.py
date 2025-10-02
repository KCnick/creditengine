import os
import joblib

MODEL_PATH = os.environ.get("MODEL_PATH", "model.joblib")

def load_model():
    return joblib.load(MODEL_PATH)
