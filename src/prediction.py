

import os
import joblib

DEPLOYMENT_THRESHOLD = 0.39

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
FAIR_MODEL_PATH = os.path.join(MODELS_DIR, "randomforest_fair.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")


def load_model():
    """
    Returns (model, is_fair_model: bool, source_path: str|None).
    Prefers the fairness-audited model if present.
    """
    if os.path.exists(FAIR_MODEL_PATH):
        return joblib.load(FAIR_MODEL_PATH), True, FAIR_MODEL_PATH
    
    return None, False, None


def load_scaler():
    """Returns the fitted StandardScaler, or None if scaler.pkl isn't there yet."""
    if os.path.exists(SCALER_PATH):
        return joblib.load(SCALER_PATH)
    return None


def predict_churn(model_input_df, model, threshold: float = DEPLOYMENT_THRESHOLD):
    """
    model_input_df: single-row DataFrame already preprocessed/scaled/aligned
    Returns: (probability: float, is_churn: bool, label: str)
    """
    probability = float(model.predict_proba(model_input_df)[0, 1])
    is_churn = probability >= threshold
    label = "Likely to Churn" if is_churn else "Likely to Stay"
    return probability, is_churn, label


def risk_bucket(probability: float) -> str:
    if probability >= 0.39:
        return "HIGH"
    elif probability >= 0.20:
        return "MEDIUM"
    return "LOW"
