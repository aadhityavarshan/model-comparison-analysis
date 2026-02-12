from src.models.baseline import train_baseline_model
from src.models.tree_model import train_rf_model

if __name__ == "__main__":
    print("=" * 60)
    print("Training Baseline Model...")
    print("=" * 60)
    baseline_model, baseline_mae = train_baseline_model()
    
    print("\n" + "=" * 60)
    print("Training RandomForest Model...")
    print("=" * 60)
    rf_model, rf_mae = train_rf_model()
    
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Baseline MAE: {baseline_mae:.2f}")
    print(f"RandomForest MAE: {rf_mae:.2f}")
    print(f"Best Model: {'RandomForest' if rf_mae < baseline_mae else 'Baseline'}")
    print("=" * 60)
