# Comparing Models

A complete end-to-end machine learning system that predicts daily restaurant demand using synthetic smart‑city data.  
You get a real ML engineering workflow: ingestion, feature engineering, model training, experiment tracking, API serving, and EDA.

---

## 📌 Project Overview
This project simulates a smart‑city food demand prediction service.  
It generates synthetic data, builds features, trains baseline and RandomForest models, tracks experiments with MLflow, and exposes predictions through a FastAPI endpoint.

It’s structured like a real production ML project, not a notebook experiment.

---

## 🚀 Features

### **Machine Learning Pipeline**
- Data ingestion (synthetic restaurant demand)
- Feature engineering (rolling averages, day-of-week, month, etc.)
- Baseline Linear Regression model
- RandomForest model with GridSearchCV
- Fully versioned models saved as `.joblib`

### **Experiment Tracking (MLflow)**
Tracks:
- Parameters
- Metrics (MAE)
- Model artifacts
- Feature sets
- RandomForest hyperparameters

Easily compare runs and choose the best model.

### **FastAPI Prediction Service**
Serve predictions via:
```
POST /predict
```

Supports selecting model:
- `"baseline"`
- `"rf"`

### **Exploratory Data Analysis**
`notebooks/eda.ipynb` includes:
- Distribution plots
- Correlation heatmap
- Rolling average visualization
- Promo and weekend effects

---

## 📁 Project Structure

```
smart-city-food-demand/
│
├── api/
│   └── main.py                 # FastAPI prediction service
│
├── data/
│   ├── raw/
│   │   └── restaurant_data.csv
│   └── processed/
│       ├── processed_restaurant_data.csv
│       ├── baseline_model.joblib
│       └── rf_model.joblib
│
├── notebooks/
│   └── eda.ipynb              # Exploratory Data Analysis
│
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration (paths, constants)
│   ├── data_ingestion.py      # Data generation
│   ├── data_prep.py           # Feature engineering
│   ├── train.py               # Training orchestration
│   ├── predict.py             # Prediction logic
│   └── models/
│       ├── __init__.py
│       ├── baseline.py        # Baseline Linear Regression
│       └── tree_model.py      # RandomForest with GridSearchCV
│
├── tests/
│   ├── test_data_ingestion.py # Data ingestion tests
│   ├── test_data_prep.py      # Data preparation tests
│   ├── test_models.py         # Model training/inference tests
│   └── test_api.py            # API endpoint tests
│
├── mlruns/                    # MLflow experiment tracking (auto-generated)
├── run_pipeline.py            # Full pipeline orchestration script
├── requirements.txt           # Python dependencies
└── README.md
```

---

## 🔧 Setup Instructions

### 1. Create and activate virtual environment
```
python -m venv .venv
.\.venv\Scripts\activate
```

### 2. Install dependencies
```
pip install -r requirements.txt
```

---

## 🔄 Running the Full Pipeline

### **Quick Start (Recommended)**
Run the entire pipeline with one command:
```
python run_pipeline.py
```

This will automatically:
1. Generate synthetic data
2. Build features
3. Train baseline and RandomForest models
4. Log experiments to MLflow
5. Display a summary comparing models

### **Step-by-Step**

Alternatively, run each step individually:

#### 1. Generate synthetic data
```
python -m src.data_ingestion
```

#### 2. Build features
```
python -m src.data_prep
```

#### 3. Train models (baseline + RF)
```
python -m src.train
```

Models produced:
```
data/processed/baseline_model.joblib
data/processed/rf_model.joblib
```

---

## 🧪 Running Tests

Install test dependencies (already in requirements.txt):
```
pip install -r requirements.txt
```

Run all tests:
```
pytest tests/ -v
```

Run specific test file:
```
pytest tests/test_models.py -v
```

Run with coverage:
```
pytest tests/ --cov=src --cov-report=html
```

**Test Modules:**
- `test_data_ingestion.py` — Data generation and validation
- `test_data_prep.py` — Feature engineering tests
- `test_models.py` — Model training and prediction tests
- `test_api.py` — FastAPI endpoint tests

---

## ⚙️ Running the API

Start the FastAPI server:
```
uvicorn api.main:app --reload
```

Open interactive docs:
```
http://127.0.0.1:8000/docs
```

### Example Request
```json
{
  "temp_c": 22,
  "is_weekend": 1,
  "promo": 1,
  "rolling_orders_7d": 85,
  "month": 5,
  "day_of_week": 6,
  "model_name": "rf"
}
```

### Example Response
```json
{
  "predicted_orders": 94.21
}
```

---

## 📊 Custom Prediction Dashboard

Instead of using the default API docs, use the interactive Streamlit dashboard:

### Start the dashboard
```
streamlit run dashboard.py
```

Opens automatically at:
```
http://localhost:8501
```

### Features
- 🎨 Intuitive form for all input parameters
- 📈 Real-time prediction display
- 🔄 Toggle between Baseline and RandomForest models
- 🚀 Visual feedback and error handling
- 📝 Built-in instructions and help text

**Note:** The API server must be running (`uvicorn api.main:app --reload`) for the dashboard to work.

---

## 📊 MLflow Tracking

Start MLflow:
```
mlflow ui
```

Visit:
```
http://127.0.0.1:5000
```

You’ll see:
- all runs
- parameters
- metrics
- comparison graphs
- stored models

This gives you a real experiment dashboard.

---

## 🧠 EDA Notebook

Run:
```
jupyter notebook
```

Open:
```
notebooks/eda.ipynb
```

The notebook covers:
- time-series demand
- promo impact
- weekend vs weekday
- correlation heatmaps
- rolling averages

---

## 🛠 Future Roadmap

- Add XGBoost or LightGBM
- Add Optuna hyperparameter tuning
- Build Streamlit dashboard for real-time predictions
- Containerize with Docker
- Deploy API to Render / Railway / AWS
- Add Airflow pipeline for automated retraining
- Add MLflow Model Registry

---

## 🙌 Author
Smart City Food Demand Prediction – full ML engineering pipeline demo.
