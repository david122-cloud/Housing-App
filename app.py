import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- 1. Load Assets (Model & Scaler) ---
@st.cache_data
def load_assets():
    # We use 'rb' (read binary) because these are pickle files
    with open('lasso_model.pkl', 'rb') as file:
        model = pickle.load(file)
    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return model, scaler

# Load them now. If this fails, it means the files aren't in the folder!
try:
    lasso_model, scaler = load_assets()
except FileNotFoundError:
    st.error("Error: Files not found! Make sure 'lasso_model.pkl' and 'scaler.pkl' are in the same folder as this app.py file.")
    st.stop()

# --- 2. App Interface ---
st.title("🏡 California House Price Predictor")
st.markdown("Enter the features below to predict the median house value.")

st.header("Input House Features:")

def user_input_features():
    # --- Longitude Fix ---
    # User sees a POSITIVE number (e.g., 122.23)
    # We force min_value=0 so they cannot type a negative number
    longitude_input = st.number_input('Longitude (e.g., 122.23)', value=122.23, min_value=0.0, step=0.01)
    
    # Model gets a NEGATIVE number (e.g., -122.23) because California is West
    longitude = -abs(longitude_input)

    # --- Other Features ---
    latitude = st.number_input('Latitude (e.g., 37.88)', value=37.88, step=0.01)
    housing_median_age = st.slider('Housing Median Age', 1, 52, 28)
    
    # We add min_value=0 to these to prevent impossible negative rooms/people
    total_rooms = st.number_input('Total Rooms', value=2000, step=100, min_value=1)
    total_bedrooms = st.number_input('Total Bedrooms', value=400, step=50, min_value=1)
    population = st.number_input('Population', value=1000, step=100, min_value=1)
    households = st.number_input('Households', value=380, step=50, min_value=1)
    median_income = st.number_input('Median Income (in $10k)', value=4.0, step=0.1, min_value=0.0)
    
    # We save these in a dictionary matching the training data structure
    data = {
        'longitude': longitude,
        'latitude': latitude,
        'housing_median_age': housing_median_age,
        'total_rooms': total_rooms,
        'total_bedrooms': total_bedrooms,
        'population': population,
        'households': households,
        'median_income': median_income,
        'median_house_value': 0 # Placeholder (we drop this later)
    }
    return pd.DataFrame(data, index=[0])

# Capture the input
input_df = user_input_features()

# --- 3. Prediction Logic ---
if st.button('Predict House Price'):
    # A. Feature Engineering (Crucial Step: Re-creating the Ratios)
    input_df['rooms_per_household'] = input_df['total_rooms'] / input_df['households']
    input_df['bedrooms_per_room'] = input_df['total_bedrooms'] / input_df['total_rooms']
    input_df['population_per_household'] = input_df['population'] / input_df['households']

    # B. Prepare for Model (Drop the placeholder target)
    X_new = input_df.drop('median_house_value', axis=1)

    # C. Scale the Data (Using the scaler we loaded)
    X_new_scaled = scaler.transform(X_new)
    
    # D. Predict (Using the model we loaded)
    prediction = lasso_model.predict(X_new_scaled)
    
    # E. Show Result
    st.success("Prediction Complete!")
    st.balloons()
    st.header(f"Estimated Value: **${prediction[0]:,.2f}**")