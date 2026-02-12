import pytest
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from src.data_ingestion import generate_dummy_data, save_raw_data
from src.config import RAW_DATA_DIR


class TestDataIngestion:
    """Test data ingestion module."""

    def test_generate_dummy_data_shape(self):
        """Test that generated data has correct shape."""
        n_days = 120
        df = generate_dummy_data(n_days=n_days)
        assert len(df) == n_days
        assert df.shape[1] == 5  # date, temp_c, is_weekend, promo, orders

    def test_generate_dummy_data_columns(self):
        """Test that all required columns are present."""
        df = generate_dummy_data()
        required_cols = ["date", "temp_c", "is_weekend", "promo", "orders"]
        assert all(col in df.columns for col in required_cols)

    def test_generate_dummy_data_types(self):
        """Test that columns have correct data types."""
        df = generate_dummy_data()
        assert pd.api.types.is_datetime64_any_dtype(df["date"])
        assert pd.api.types.is_numeric_dtype(df["temp_c"])
        assert pd.api.types.is_numeric_dtype(df["is_weekend"])
        assert pd.api.types.is_numeric_dtype(df["promo"])
        assert pd.api.types.is_numeric_dtype(df["orders"])

    def test_generate_dummy_data_values(self):
        """Test that generated values are within expected ranges."""
        df = generate_dummy_data()
        
        # is_weekend should be 0 or 1
        assert df["is_weekend"].isin([0, 1]).all()
        
        # promo should be 0 or 1
        assert df["promo"].isin([0, 1]).all()
        
        # orders should be positive
        assert (df["orders"] > 0).all()
        
        # temperature should be within reasonable range (generated from N(20, 5))
        assert df["temp_c"].min() > 0
        assert df["temp_c"].max() < 50

    def test_generate_dummy_data_dates(self):
        """Test that dates are sequential."""
        n_days = 30
        df = generate_dummy_data(n_days=n_days)
        date_diffs = df["date"].diff().dt.days[1:]
        assert (date_diffs == 1).all()

    def test_save_raw_data_creates_file(self, tmp_path, monkeypatch):
        """Test that save_raw_data creates a CSV file."""
        # Mock RAW_DATA_DIR to use tmp_path
        monkeypatch.setattr("src.data_ingestion.RAW_DATA_DIR", tmp_path)
        
        save_raw_data()
        
        output_file = tmp_path / "restaurant_data.csv"
        assert output_file.exists()
        
        # Verify file contents
        df = pd.read_csv(output_file)
        assert len(df) == 120  # default is 120 days
        assert all(col in df.columns for col in ["date", "temp_c", "is_weekend", "promo", "orders"])

    def test_save_raw_data_file_is_valid_csv(self, tmp_path, monkeypatch):
        """Test that saved file is a valid CSV."""
        monkeypatch.setattr("src.data_ingestion.RAW_DATA_DIR", tmp_path)
        
        save_raw_data()
        
        output_file = tmp_path / "restaurant_data.csv"
        df = pd.read_csv(output_file)
        
        # Should be able to parse and have no missing values
        assert df.isnull().sum().sum() == 0
