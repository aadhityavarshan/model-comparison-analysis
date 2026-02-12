import joblib
import numpy as np
from datetime import datetime
from .config import PROCESSED_DATA_DIR

def load_model(model_name: str = "baseline"):
    if model_name == "rf":
        path = PROCESSED_DATA_DIR / "rf_model.joblib"
    else:
        path = PROCESSED_DATA_DIR / "baseline_model.joblib"
    return joblib.load(path)

def predict_one(temp_c, is_weekend, promo, rolling_orders_7d, month, day_of_week, model_name: str = "baseline"):
    model = load_model(model_name)
    features = np.array([[temp_c, is_weekend, promo, day_of_week, rolling_orders_7d, month]])
    prediction = model.predict(features)
    return prediction[0]