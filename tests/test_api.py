import pytest
from fastapi.testclient import TestClient
import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from pathlib import Path

from api.main import app, PredictionRequest, PredictionResponse
from src.config import PROCESSED_DATA_DIR


client = TestClient(app)


class TestAPIEndpoints:
    """Test FastAPI prediction endpoints."""

    @pytest.fixture(autouse=True)
    def setup_models(self, tmp_path, monkeypatch):
        """Setup baseline and RF models for testing."""
        monkeypatch.setattr("src.predict.PROCESSED_DATA_DIR", tmp_path)
        
        # Create and save baseline model
        baseline_model = LinearRegression()
        X = np.array([[20, 0, 0, 0, 100, 1], [25, 1, 1, 5, 110, 6]])
        y = np.array([80, 120])
        baseline_model.fit(X, y)
        baseline_path = tmp_path / "baseline_model.joblib"
        joblib.dump(baseline_model, baseline_path)
        
        # Create and save RF model
        rf_model = RandomForestRegressor(n_estimators=2, random_state=42)
        rf_model.fit(X, y)
        rf_path = tmp_path / "rf_model.joblib"
        joblib.dump(rf_model, rf_path)

    def test_predict_endpoint_exists(self):
        """Test that /predict endpoint exists."""
        response = client.post("/predict", json={
            "temp_c": 22,
            "is_weekend": 1,
            "promo": 1,
            "rolling_orders_7d": 85,
            "month": 5,
            "day_of_week": 6,
            "model_name": "baseline"
        })
        assert response.status_code == 200

    def test_predict_endpoint_with_baseline_model(self):
        """Test prediction endpoint with baseline model."""
        response = client.post("/predict", json={
            "temp_c": 22,
            "is_weekend": 1,
            "promo": 1,
            "rolling_orders_7d": 85,
            "month": 5,
            "day_of_week": 6,
            "model_name": "baseline"
        })
        assert response.status_code == 200
        data = response.json()
        assert "predicted_orders" in data
        assert isinstance(data["predicted_orders"], (int, float))
        assert data["predicted_orders"] > 0

    def test_predict_endpoint_with_rf_model(self):
        """Test prediction endpoint with RandomForest model."""
        response = client.post("/predict", json={
            "temp_c": 22,
            "is_weekend": 1,
            "promo": 1,
            "rolling_orders_7d": 85,
            "month": 5,
            "day_of_week": 6,
            "model_name": "rf"
        })
        assert response.status_code == 200
        data = response.json()
        assert "predicted_orders" in data
        assert isinstance(data["predicted_orders"], (int, float))

    def test_predict_endpoint_default_model_is_baseline(self):
        """Test that default model is baseline when not specified."""
        response = client.post("/predict", json={
            "temp_c": 22,
            "is_weekend": 1,
            "promo": 1,
            "rolling_orders_7d": 85,
            "month": 5,
            "day_of_week": 6
        })
        assert response.status_code == 200
        data = response.json()
        assert "predicted_orders" in data

    def test_predict_endpoint_response_schema(self):
        """Test that response follows PredictionResponse schema."""
        response = client.post("/predict", json={
            "temp_c": 22.5,
            "is_weekend": 0,
            "promo": 1,
            "rolling_orders_7d": 95.5,
            "month": 3,
            "day_of_week": 2
        })
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        assert len(data) == 1
        assert "predicted_orders" in data

    def test_predict_endpoint_various_inputs(self):
        """Test prediction with various input values."""
        test_cases = [
            {"temp_c": 10, "is_weekend": 0, "promo": 0, "rolling_orders_7d": 50, "month": 1, "day_of_week": 0},
            {"temp_c": 35, "is_weekend": 1, "promo": 1, "rolling_orders_7d": 150, "month": 7, "day_of_week": 6},
            {"temp_c": 20, "is_weekend": 1, "promo": 0, "rolling_orders_7d": 100, "month": 4, "day_of_week": 5},
        ]
        
        for case in test_cases:
            response = client.post("/predict", json=case)
            assert response.status_code == 200
            assert "predicted_orders" in response.json()

    def test_predict_endpoint_missing_fields(self):
        """Test that endpoint validates required fields."""
        incomplete_request = {
            "temp_c": 22,
            "is_weekend": 1,
            # Missing other fields
        }
        response = client.post("/predict", json=incomplete_request)
        assert response.status_code == 422  # Unprocessable Entity

    def test_predict_returns_numeric_prediction(self):
        """Test that prediction is a valid number."""
        response = client.post("/predict", json={
            "temp_c": 22,
            "is_weekend": 1,
            "promo": 1,
            "rolling_orders_7d": 85,
            "month": 5,
            "day_of_week": 6,
            "model_name": "baseline"
        })
        assert response.status_code == 200
        data = response.json()
        prediction = data["predicted_orders"]
        assert not np.isnan(prediction)
        assert not np.isinf(prediction)


class TestAPIRequestModel:
    """Test PredictionRequest model validation."""

    def test_prediction_request_valid(self):
        """Test that valid request passes validation."""
        request = PredictionRequest(
            temp_c=22,
            is_weekend=1,
            promo=1,
            rolling_orders_7d=85,
            month=5,
            day_of_week=6,
            model_name="baseline"
        )
        assert request.temp_c == 22
        assert request.is_weekend == 1
        assert request.model_name == "baseline"

    def test_prediction_response_valid(self):
        """Test that valid response passes validation."""
        response = PredictionResponse(predicted_orders=95.5)
        assert response.predicted_orders == 95.5
