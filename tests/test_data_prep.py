import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.data_prep import load_raw_data, build_features, save_processed
from src.config import PROCESSED_DATA_DIR


class TestDataPrep:
    """Test data preparation module."""

    @pytest.fixture
    def sample_raw_data(self):
        """Create sample raw data for testing."""
        dates = pd.date_range(start="2024-01-01", periods=30, freq="D")
        return pd.DataFrame({
            "date": dates,
            "temp_c": np.random.normal(20, 5, 30),
            "is_weekend": [1 if d.weekday() >= 5 else 0 for d in dates],
            "promo": np.random.randint(0, 2, 30),
            "orders": np.random.randint(50, 150, 30)
        })

    def test_build_features_adds_day_of_week(self, sample_raw_data):
        """Test that day_of_week is added correctly."""
        df = build_features(sample_raw_data.copy())
        
        assert "day_of_week" in df.columns
        # Check that day_of_week matches actual weekday
        for idx, row in df.iterrows():
            assert row["day_of_week"] == row["date"].weekday()

    def test_build_features_adds_month(self, sample_raw_data):
        """Test that month is added correctly."""
        df = build_features(sample_raw_data.copy())
        
        assert "month" in df.columns
        # Check that month matches actual month
        for idx, row in df.iterrows():
            assert row["month"] == row["date"].month

    def test_build_features_adds_rolling_average(self, sample_raw_data):
        """Test that rolling 7-day average is calculated."""
        df = build_features(sample_raw_data.copy())
        
        assert "rolling_orders_7d" in df.columns
        
        # First row should have only 1 value (min_periods=1)
        assert df.iloc[0]["rolling_orders_7d"] == df.iloc[0]["orders"]
        
        # 7th row should be average of first 7
        expected_7th = df.iloc[:7]["orders"].mean()
        assert abs(df.iloc[6]["rolling_orders_7d"] - expected_7th) < 0.01

    def test_build_features_preserves_original_columns(self, sample_raw_data):
        """Test that original columns are preserved."""
        original_cols = set(sample_raw_data.columns)
        df = build_features(sample_raw_data.copy())
        
        # All original columns should still exist
        assert original_cols.issubset(set(df.columns))

    def test_build_features_no_missing_values(self, sample_raw_data):
        """Test that no NaN values are introduced (with min_periods=1)."""
        df = build_features(sample_raw_data.copy())
        
        # rolling_orders_7d should not be NaN due to min_periods=1
        assert df["rolling_orders_7d"].isnull().sum() == 0

    def test_build_features_sorts_by_date(self):
        """Test that data is sorted by date."""
        # Create unsorted data
        dates = [
            pd.Timestamp("2024-01-05"),
            pd.Timestamp("2024-01-01"),
            pd.Timestamp("2024-01-03"),
            pd.Timestamp("2024-01-02"),
        ]
        df = pd.DataFrame({
            "date": dates,
            "temp_c": [20, 21, 22, 23],
            "is_weekend": [0, 0, 0, 0],
            "promo": [0, 1, 0, 1],
            "orders": [100, 110, 105, 115]
        })
        
        df_processed = build_features(df)
        
        # Check that dates are sorted
        assert (df_processed["date"].diff().dt.days[1:] >= 0).all()

    def test_save_processed_creates_file(self, sample_raw_data, tmp_path, monkeypatch):
        """Test that save_processed creates a CSV file."""
        monkeypatch.setattr("src.data_prep.PROCESSED_DATA_DIR", tmp_path)
        
        save_processed(sample_raw_data)
        
        output_file = tmp_path / "processed_restaurant_data.csv"
        assert output_file.exists()

    def test_save_processed_file_is_readable(self, sample_raw_data, tmp_path, monkeypatch):
        """Test that saved file can be read back."""
        monkeypatch.setattr("src.data_prep.PROCESSED_DATA_DIR", tmp_path)
        
        save_processed(sample_raw_data)
        
        output_file = tmp_path / "processed_restaurant_data.csv"
        df_read = pd.read_csv(output_file)
        
        assert len(df_read) == len(sample_raw_data)
        assert list(df_read.columns) == list(sample_raw_data.columns)
