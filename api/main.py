from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
from src.predict import predict_one

app = FastAPI(title="Food Demand Prediction API")

class PredictionRequest(BaseModel):
    temp_c: float
    is_weekend: int
    promo: int
    rolling_orders_7d: float
    month: int
    day_of_week: int

class PredictionResponse(BaseModel):
    predicted_orders: float
    timestamp: datetime

@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    y = predict_one(
        temp_c=request.temp_c,
        is_weekend=request.is_weekend,
        promo=request.promo,
        rolling_orders_7d=request.rolling_orders_7d,
        month=request.month,
        day_of_week=request.day_of_week
    )
    return PredictionResponse(
        predicted_orders=y,
        timestamp=datetime.now(datetime.timezone.utc)
    )