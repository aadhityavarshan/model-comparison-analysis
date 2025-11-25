import joblib
import numpy as np
from datetime import datetime
from .config import PROCESSED_DATA_DIR

def load_model():
    return joblib.load(PROCESSED_DATA_DIR / "baseline_model.joblib")

def predict_one(temp_c, is_weekend, promo, rolling_orders_7d, month, day_of_week):
    model = load_model()
    features = np.array([[temp_c, is_weekend, promo, day_of_week, rolling_orders_7d, month]])
    prediction = model.predict(features)
    return prediction[0]