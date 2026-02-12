import pytest
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from src.models.baseline import train_baseline_model
from src.models.tree_model import train_rf_model
from src.predict import load_model, predict_one
from src.config import PROCESSED_DATA_DIR


class TestBaselineModel:
    """Test baseline model training and inference."""

    def test_train_baseline_model_returns_tuple(self, tmp_path, monkeypatch):
        """Test that train_baseline_model returns a tuple of (model, mae)."""
        monkeypatch.setattr("src.models.baseline.PROCESSED_DATA_DIR", tmp_path)
        
        # Create dummy processed data
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        dummy_data = pd.DataFrame({
            "date": dates,
            "temp_c": np.random.normal(20, 5, 100),
            "is_weekend": [1 if d.weekday() >= 5 else 0 for d in dates],
            "promo": np.random.randint(0, 2, 100),
            "day_of_week": [d.weekday() for d in dates],
            "rolling_orders_7d": np.random.normal(100, 20, 100),
            "month": [d.month for d in dates],
            "orders": np.random.randint(50, 150, 100)
        })
        
        processed_file = tmp_path / "processed_restaurant_data.csv"
        dummy_data.to_csv(processed_file, index=False)
        
        model, mae = train_baseline_model()
        
        assert isinstance(model, LinearRegression)
        assert isinstance(mae, (float, np.floating))
        assert mae > 0

    def test_train_baseline_model_saves_joblib(self, tmp_path, monkeypatch):
        """Test that baseline model is saved as joblib file."""
        monkeypatch.setattr("src.models.baseline.PROCESSED_DATA_DIR", tmp_path)
        
        # Create dummy processed data
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        dummy_data = pd.DataFrame({
            "date": dates,
            "temp_c": np.random.normal(20, 5, 100),
            "is_weekend": [1 if d.weekday() >= 5 else 0 for d in dates],
            "promo": np.random.randint(0, 2, 100),
            "day_of_week": [d.weekday() for d in dates],
            "rolling_orders_7d": np.random.normal(100, 20, 100),
            "month": [d.month for d in dates],
            "orders": np.random.randint(50, 150, 100)
        })
        
        processed_file = tmp_path / "processed_restaurant_data.csv"
        dummy_data.to_csv(processed_file, index=False)
        
        train_baseline_model()
        
        model_file = tmp_path / "baseline_model.joblib"
        assert model_file.exists()
        
        # Load and verify
        loaded_model = joblib.load(model_file)
        assert isinstance(loaded_model, LinearRegression)


class TestRandomForestModel:
    """Test RandomForest model training and inference."""

    def test_train_rf_model_returns_tuple(self, tmp_path, monkeypatch):
        """Test that train_rf_model returns a tuple of (model, mae)."""
        monkeypatch.setattr("src.models.tree_model.PROCESSED_DATA_DIR", tmp_path)
        
        # Create dummy processed data
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        dummy_data = pd.DataFrame({
            "date": dates,
            "temp_c": np.random.normal(20, 5, 100),
            "is_weekend": [1 if d.weekday() >= 5 else 0 for d in dates],
            "promo": np.random.randint(0, 2, 100),
            "day_of_week": [d.weekday() for d in dates],
            "rolling_orders_7d": np.random.normal(100, 20, 100),
            "month": [d.month for d in dates],
            "orders": np.random.randint(50, 150, 100)
        })
        
        processed_file = tmp_path / "processed_restaurant_data.csv"
        dummy_data.to_csv(processed_file, index=False)
        
        model, mae = train_rf_model()
        
        assert isinstance(model, RandomForestRegressor)
        assert isinstance(mae, (float, np.floating))
        assert mae > 0

    def test_train_rf_model_saves_joblib(self, tmp_path, monkeypatch):
        """Test that RF model is saved as joblib file."""
        monkeypatch.setattr("src.models.tree_model.PROCESSED_DATA_DIR", tmp_path)
        
        # Create dummy processed data
        dates = pd.date_range(start="2024-01-01", periods=100, freq="D")
        dummy_data = pd.DataFrame({
            "date": dates,
            "temp_c": np.random.normal(20, 5, 100),
            "is_weekend": [1 if d.weekday() >= 5 else 0 for d in dates],
            "promo": np.random.randint(0, 2, 100),
            "day_of_week": [d.weekday() for d in dates],
            "rolling_orders_7d": np.random.normal(100, 20, 100),
            "month": [d.month for d in dates],
            "orders": np.random.randint(50, 150, 100)
        })
        
        processed_file = tmp_path / "processed_restaurant_data.csv"
        dummy_data.to_csv(processed_file, index=False)
        
        train_rf_model()
        
        model_file = tmp_path / "rf_model.joblib"
        assert model_file.exists()
        
        # Load and verify
        loaded_model = joblib.load(model_file)
        assert isinstance(loaded_model, RandomForestRegressor)


class TestPrediction:
    """Test prediction module."""

    def test_predict_one_returns_float(self, tmp_path, monkeypatch):
        """Test that predict_one returns a float."""
        monkeypatch.setattr("src.predict.PROCESSED_DATA_DIR", tmp_path)
        
        # Create a simple baseline model
        model = LinearRegression()
        X = np.array([[20, 0, 0, 0, 100, 5]])
        y = np.array([80])
        model.fit(X, y)
        
        model_file = tmp_path / "baseline_model.joblib"
        joblib.dump(model, model_file)
        
        prediction = predict_one(
            temp_c=22,
            is_weekend=1,
            promo=1,
            rolling_orders_7d=85,
            month=5,
            day_of_week=6,
            model_name="baseline"
        )
        
        assert isinstance(prediction, (float, np.floating))

    def test_predict_one_baseline_model(self, tmp_path, monkeypatch):
        """Test prediction with baseline model."""
        monkeypatch.setattr("src.predict.PROCESSED_DATA_DIR", tmp_path)
        
        # Create a simple baseline model with known output
        model = LinearRegression()
        X = np.array([[20, 0, 0, 0, 100, 1], [25, 1, 1, 5, 110, 6]])
        y = np.array([80, 120])
        model.fit(X, y)
        
        model_file = tmp_path / "baseline_model.joblib"
        joblib.dump(model, model_file)
        
        prediction = predict_one(
            temp_c=22,
            is_weekend=1,
            promo=1,
            rolling_orders_7d=85,
            month=5,
            day_of_week=3,
            model_name="baseline"
        )
        
        assert prediction > 0

    def test_predict_one_rf_model(self, tmp_path, monkeypatch):
        """Test prediction with RandomForest model."""
        monkeypatch.setattr("src.predict.PROCESSED_DATA_DIR", tmp_path)
        
        # Create a simple RF model
        model = RandomForestRegressor(n_estimators=2, random_state=42)
        X = np.array([[20, 0, 0, 0, 100, 1], [25, 1, 1, 5, 110, 6]])
        y = np.array([80, 120])
        model.fit(X, y)
        
        model_file = tmp_path / "rf_model.joblib"
        joblib.dump(model, model_file)
        
        prediction = predict_one(
            temp_c=22,
            is_weekend=1,
            promo=1,
            rolling_orders_7d=85,
            month=5,
            day_of_week=3,
            model_name="rf"
        )
        
        assert prediction > 0

    def test_load_model_baseline(self, tmp_path, monkeypatch):
        """Test loading baseline model."""
        monkeypatch.setattr("src.predict.PROCESSED_DATA_DIR", tmp_path)
        
        model = LinearRegression()
        model_file = tmp_path / "baseline_model.joblib"
        joblib.dump(model, model_file)
        
        loaded = load_model("baseline")
        assert isinstance(loaded, LinearRegression)

    def test_load_model_rf(self, tmp_path, monkeypatch):
        """Test loading RandomForest model."""
        monkeypatch.setattr("src.predict.PROCESSED_DATA_DIR", tmp_path)
        
        model = RandomForestRegressor()
        model_file = tmp_path / "rf_model.joblib"
        joblib.dump(model, model_file)
        
        loaded = load_model("rf")
        assert isinstance(loaded, RandomForestRegressor)
