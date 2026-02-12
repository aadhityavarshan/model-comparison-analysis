#!/usr/bin/env python
"""
Pipeline orchestration script for the smart-city food demand prediction system.

This script runs the complete ML pipeline end-to-end:
1. Data ingestion (generate synthetic data)
2. Data preparation (feature engineering)
3. Model training (baseline and RandomForest)

Usage:
    python run_pipeline.py
"""

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

from src.data_ingestion import save_raw_data
from src.data_prep import load_raw_data, build_features, save_processed
from src.models.baseline import train_baseline_model
from src.models.tree_model import train_rf_model


def print_header(text: str):
    """Print a styled header."""
    print("\n" + "=" * 70)
    print(f"  {text}")
    print("=" * 70 + "\n")


def run_pipeline():
    """Run the complete ML pipeline."""
    
    try:
        # Step 1: Data Ingestion
        print_header("STEP 1: DATA INGESTION")
        print("Generating synthetic restaurant demand data...")
        save_raw_data()
        print("✓ Raw data generated successfully")

        # Step 2: Data Preparation
        print_header("STEP 2: DATA PREPARATION")
        print("Loading raw data...")
        raw_df = load_raw_data()
        print(f"✓ Loaded {len(raw_df)} records")
        
        print("Building features (day_of_week, month, rolling_orders_7d)...")
        processed_df = build_features(raw_df)
        
        print("Saving processed data...")
        save_processed(processed_df)
        print("✓ Data preparation completed")

        # Step 3: Model Training
        print_header("STEP 3: MODEL TRAINING")
        
        print("Training Baseline Linear Regression model...")
        baseline_model, baseline_mae = train_baseline_model()
        print(f"✓ Baseline model trained - MAE: {baseline_mae:.2f}")
        
        print("\nTraining RandomForest model (with GridSearchCV)...")
        rf_model, rf_mae = train_rf_model()
        print(f"✓ RandomForest model trained - MAE: {rf_mae:.2f}")

        # Summary
        print_header("PIPELINE COMPLETED SUCCESSFULLY")
        print(f"Baseline Model MAE:     {baseline_mae:.2f}")
        print(f"RandomForest Model MAE: {rf_mae:.2f}")
        
        best_model = "RandomForest" if rf_mae < baseline_mae else "Baseline"
        improvement = abs(rf_mae - baseline_mae)
        print(f"\n🏆 Best Model: {best_model} (improvement: {improvement:.2f})")
        
        print("\n📊 Next Steps:")
        print("   1. View experiment results: mlflow ui")
        print("   2. Start the API server: uvicorn api.main:app --reload")
        print("   3. Test predictions: http://127.0.0.1:8000/docs")
        print("   4. Run tests: pytest tests/")
        print("\n" + "=" * 70 + "\n")
        
        return True
        
    except Exception as e:
        print_header("PIPELINE FAILED")
        print(f"❌ Error: {str(e)}")
        print("\nTraceback:")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_pipeline()
    sys.exit(0 if success else 1)
