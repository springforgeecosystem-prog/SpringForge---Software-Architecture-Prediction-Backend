#CodeGeneration-backend
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd

app = FastAPI(
    title="SpringForge Architecture Classifier API",
    description="Predicts MVC / Layered / Clean architecture using tuned Random Forest.",
    version="1.0.0"
)

# ------------------------------------------------------------
# Load artifacts
# ------------------------------------------------------------

model = joblib.load("architecture_model/best_model_RF_tuned.pkl")
feature_list = joblib.load("architecture_model/feature_list.pkl")

# Try to load label encoder (if the model uses encoded labels)
try:
    label_encoder = joblib.load("architecture_model/label_encoder_xgb.pkl")
    has_label_encoder = True
except:
    label_encoder = None
    has_label_encoder = False


# ------------------------------------------------------------
# Schema for incoming POST request
# ------------------------------------------------------------

class ArchitectureInput(BaseModel):
    data: dict   # All feature:value pairs


# ------------------------------------------------------------
# Prediction helper
# ------------------------------------------------------------

def prepare_input(sample_dict: dict):
    """Convert dict -> ordered feature vector required by the model."""
    vector = np.array([sample_dict.get(f, 0) for f in feature_list]).reshape(1, -1)
    return vector


def predict_arch(sample_dict: dict):
    x = prepare_input(sample_dict)

    pred_raw = model.predict(x)[0]

    # If integer label & label encoder exists → decode
    if has_label_encoder:
        if isinstance(pred_raw, (int, np.integer)):
            pred = label_encoder.inverse_transform([pred_raw])[0]
        else:
            pred = pred_raw
    else:
        pred = pred_raw

    # Probability distribution
    probs = model.predict_proba(x)[0]
    prob_dict = {
        (label_encoder.classes_[i] if has_label_encoder else model.classes_[i]): float(probs[i])
        for i in range(len(probs))
    }

    confidence = float(np.max(probs))

    return pred, confidence, prob_dict


# ------------------------------------------------------------
# FastAPI endpoints
# ------------------------------------------------------------

@app.post("/predict")
def classify(payload: ArchitectureInput):
    predicted, confidence, probabilities = predict_arch(payload.data)

    return {
        "predicted_architecture": predicted,
        "confidence": confidence,
        "probabilities": probabilities
    }


@app.get("/")
def root():
    return {
        "service": "SpringForge Architecture Classifier API",
        "status": "running and deployed(with docker pipeline)",
        "model_loaded": "best_model_RF_tuned.pkl",
        "features_expected": len(feature_list)
    }
