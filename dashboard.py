"""
Streamlit dashboard for the Food Demand Prediction API.

Run with: streamlit run dashboard.py
"""

import streamlit as st
import requests
import json
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Food Demand Predictor",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .prediction-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        margin: 20px 0;
    }
    .prediction-value {
        font-size: 48px;
        font-weight: bold;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("🍽️ Food Demand Prediction Dashboard")
st.markdown("*Predict daily restaurant demand using real-world features*")

# API endpoint
API_URL = "http://127.0.0.1:8000/predict"

# Check API connection
def check_api_connection():
    try:
        requests.get("http://127.0.0.1:8000/docs", timeout=2)
        return True
    except:
        return False

# Main layout
col1, col2 = st.columns([1.5, 1])

with col1:
    st.subheader("📊 Input Features")
    
    with st.form("prediction_form"):
        # Temperature
        temp_c = st.slider(
            "Temperature (°C)",
            min_value=-10.0,
            max_value=50.0,
            value=22.0,
            step=0.5,
            help="Current temperature in Celsius"
        )
        
        # Weekend
        is_weekend = st.selectbox(
            "Day Type",
            options=[0, 1],
            format_func=lambda x: "Weekend 🎉" if x == 1 else "Weekday 📅",
            help="Is it a weekend?"
        )
        
        # Promotion
        promo = st.selectbox(
            "Promotion Active",
            options=[0, 1],
            format_func=lambda x: "Yes 🎁" if x == 1 else "No",
            help="Is there an active promotion?"
        )
        
        # Rolling average
        rolling_orders_7d = st.number_input(
            "7-Day Rolling Average Orders",
            min_value=10.0,
            max_value=200.0,
            value=100.0,
            step=5.0,
            help="Average orders in the last 7 days"
        )
        
        # Month
        month = st.selectbox(
            "Month",
            options=list(range(1, 13)),
            format_func=lambda x: datetime(2024, x, 1).strftime("%B"),
            help="Month of the year"
        )
        
        # Day of week
        day_of_week = st.selectbox(
            "Day of Week",
            options=list(range(7)),
            format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x],
            help="Day of the week"
        )
        
        # Model selection
        model_name = st.radio(
            "Select Model",
            options=["baseline", "rf"],
            format_func=lambda x: "Linear Regression (Baseline)" if x == "baseline" else "Random Forest",
            horizontal=True,
            help="Which model to use for prediction?"
        )
        
        # Submit button
        submitted = st.form_submit_button("🔮 Predict Demand", use_container_width=True)

with col2:
    if not check_api_connection():
        st.warning(
            """
            ⚠️ **API Not Running**
            
            Start the API server first:
            ```
            uvicorn api.main:app --reload
            ```
            """
        )
    else:
        st.success("✅ API Connected")

# Make prediction
if submitted:
    if not check_api_connection():
        st.error("❌ Cannot connect to API. Please start the server with: `uvicorn api.main:app --reload`")
    else:
        try:
            # Prepare request
            payload = {
                "temp_c": float(temp_c),
                "is_weekend": int(is_weekend),
                "promo": int(promo),
                "rolling_orders_7d": float(rolling_orders_7d),
                "month": int(month),
                "day_of_week": int(day_of_week),
                "model_name": model_name
            }
            
            # Make API request
            with st.spinner("Making prediction..."):
                response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                predicted_orders = result["predicted_orders"]
                
                # Display prediction
                st.markdown(
                    f"""
                    <div class="prediction-result">
                        <div style="font-size: 18px; opacity: 0.9;">Predicted Daily Orders</div>
                        <div class="prediction-value">{predicted_orders:.1f}</div>
                        <div style="font-size: 14px; opacity: 0.9;">Using {model_name.upper()} model</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Display feature summary
                st.markdown("### 📈 Summary")
                col_summary1, col_summary2, col_summary3 = st.columns(3)
                
                with col_summary1:
                    st.metric("Temperature", f"{temp_c:.1f}°C")
                
                with col_summary2:
                    st.metric("7-Day Avg", f"{rolling_orders_7d:.1f}")
                
                with col_summary3:
                    status = "🎁 Yes" if promo else "No"
                    st.metric("Promotion", status)
                
            else:
                st.error(f"❌ Error: {response.status_code}")
                st.code(response.text)
        
        except requests.exceptions.ConnectionError:
            st.error("❌ Cannot reach API. Make sure it's running on http://127.0.0.1:8000")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Footer with instructions
st.divider()
st.markdown("""
### 🚀 Getting Started

1. **Start the API server** (in a terminal):
   ```
   uvicorn api.main:app --reload
   ```

2. **Run this dashboard** (in another terminal):
   ```
   streamlit run dashboard.py
   ```

3. **Make predictions** using the form above!

### 📚 How It Works
- **Baseline**: Linear Regression model (simpler, faster)
- **Random Forest**: Ensemble model (more complex, potentially more accurate)
- **Features**: Temperature, Day Type, Promotion, Historical Average, Month, Day of Week
""")
