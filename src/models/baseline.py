import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

from ..config import PROCESSED_DATA_DIR
import mlflow
import mlflow.sklearn

def load_processed_data():
    return pd.read_csv(PROCESSED_DATA_DIR / "processed_restaurant_data.csv", parse_dates=["date"])

def train_baseline_model():
    df = load_processed_data()

    features = [
        "temp_c",
        "is_weekend",
        "promo",
        "day_of_week",
        "rolling_orders_7d",
        "month"
    ]

    X = df[features]
    y = df["orders"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    print(f"Mean Absolute Error on test set: {mae:.2f}")

    model_path = PROCESSED_DATA_DIR / "baseline_model.joblib"
    joblib.dump(model, model_path)
    print(f"Baseline model saved to {PROCESSED_DATA_DIR / 'baseline_model.joblib'}")

    mlflow.set_experiment("food_demand_experiment")
    with mlflow.start_run(run_name="baseline_linear_regression"):
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("features", ",".join(features))
        mlflow.log_metric("mae", mae)

        mlflow.log_artifact(str(model_path), artifact_path="models")
        mlflow.sklearn.log_model(model, artifact_path="sklearn-model")

    return model, mae

if __name__ == "__main__":
    train_baseline_model()

