import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import joblib

# --- Streamlit Page Configuration ---
# MOVED TO THE TOP: This must be the first Streamlit command
st.set_page_config(
    page_title="California Housing Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# --- Load Model Components ---
# Use st.cache_resource to load models only once
@st.cache_resource
def load_model_components():
    """
    Loads the saved preprocessor, model, and y_scaler.
    """
    try:
        preprocessor = joblib.load('preprocessor.joblib')
        model = keras.models.load_model('housing_model.keras')
        y_scaler = joblib.load('y_scaler.joblib')
        return preprocessor, model, y_scaler
    except FileNotFoundError:
        st.error("Model files not found! Please run 'train_and_save.py' first.")
        return None, None, None

preprocessor, model, y_scaler = load_model_components()

# --- App Title ---
st.title("🏠 California Housing Price Predictor")
st.markdown("Enter the details of a housing block to get a price prediction.")

# --- Sidebar for User Inputs ---
st.sidebar.header("Input House Features")

# Create sliders and number inputs for all 8 features
longitude = st.sidebar.slider("Longitude", -124.0, -114.0, -122.23)
latitude = st.sidebar.slider("Latitude", 32.0, 42.0, 37.88)
housing_median_age = st.sidebar.slider("Housing Median Age", 1.0, 52.0, 41.0)
total_rooms = st.sidebar.number_input("Total Rooms", min_value=1.0, value=880.0)
total_bedrooms = st.sidebar.number_input("Total Bedrooms", min_value=1.0, value=129.0)
population = st.sidebar.number_input("Population", min_value=1.0, value=322.0)
households = st.sidebar.number_input("Households", min_value=1.0, value=126.0)
median_income = st.sidebar.slider("Median Income (in $10,000s)", 0.5, 15.0, 8.3252)

# Categorical feature
ocean_proximity = st.sidebar.selectbox(
    "Ocean Proximity",
    ['NEAR BAY', '<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'ISLAND']
)

# --- Prediction Logic ---
if st.button("Predict House Value", type="primary"):
    if preprocessor is not None and model is not None and y_scaler is not None:
        # 1. Create a DataFrame from the inputs (must match the training columns)
        input_data = {
            'longitude': [longitude],
            'latitude': [latitude],
            'housing_median_age': [housing_median_age],
            'total_rooms': [total_rooms],
            'total_bedrooms': [total_bedrooms],
            'population': [population],
            'households': [households],
            'median_income': [median_income],
            'ocean_proximity': [ocean_proximity]
        }
        input_df = pd.DataFrame(input_data)
        
        st.subheader("Input Features:")
        st.dataframe(input_df)

        # 2. Preprocess the input data
        try:
            input_processed = preprocessor.transform(input_df)
            
            # 3. Make a (scaled) prediction
            # Use verbose=0 to prevent Keras logs in Streamlit
            prediction_scaled = model.predict(input_processed, verbose=0) 
            
            # 4. Inverse-transform the prediction to get actual dollars
            prediction_dollars = y_scaler.inverse_transform(prediction_scaled)
            
            # 5. Display the result
            st.subheader("Prediction:")
            st.success(f"**Predicted Median House Value: ${prediction_dollars[0][0]:,.2f}**")
        
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
    else:
        st.error("Model components are not loaded. Please ensure all model files are present.")

