
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Load model artifacts
@st.cache_resource
def load_model():
    return joblib.load('paris_housing_model.pkl')

def main():
    st.set_page_config(
        page_title="Paris Housing Price Predictor", 
        page_icon="🏠", 
        layout="wide"
    )
    
    st.title("🏠 Paris Housing Price Predictor")
    st.markdown("### Predict housing prices in Paris using machine learning")
    
    # Load model
    try:
        artifacts = load_model()
        model = artifacts['model']
        scaler = artifacts['scaler']
        feature_columns = artifacts['feature_columns']
        model_name = artifacts['model_name']
        
        st.success(f"✅ Model loaded: {model_name}")
    except FileNotFoundError:
        st.error("❌ Model file not found. Please ensure 'paris_housing_model.pkl' is in the same directory.")
        return
    
    # Sidebar for inputs
    st.sidebar.header("🏡 Property Details")
    
    # Basic property information
    property_type = st.sidebar.selectbox("Property Type", 
                                       ["apartment", "house", "studio"])
    
    arrondissement = st.sidebar.selectbox("Arrondissement", 
                                        [f"75{i:03d}" for i in range(1, 21)])
    
    bedrooms = st.sidebar.slider("Number of Bedrooms", 1, 5, 2)
    bathrooms = st.sidebar.slider("Number of Bathrooms", 1, 3, 1)
    area_sqm = st.sidebar.slider("Area (Square Meters)", 20, 200, 75)
    floor = st.sidebar.slider("Floor", 0, 10, 3)
    
    # Amenities
    st.sidebar.subheader("✨ Amenities")
    has_elevator = st.sidebar.checkbox("Has Elevator")
    has_balcony = st.sidebar.checkbox("Has Balcony")
    has_parking = st.sidebar.checkbox("Has Parking")
    
    # Additional features
    st.sidebar.subheader("📍 Location & Features")
    year_built = st.sidebar.slider("Year Built", 1950, 2024, 1990)
    distance_to_metro = st.sidebar.slider("Distance to Metro (km)", 0.1, 2.0, 0.5)
    schools_nearby = st.sidebar.slider("Schools Nearby", 0, 10, 3)
    
    # Create prediction button
    if st.sidebar.button("🔮 Predict Price", type="primary"):
        
        # Prepare input data
        input_data = {
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'area_sqm': area_sqm,
            'floor': floor,
            'has_elevator': int(has_elevator),
            'has_balcony': int(has_balcony),
            'has_parking': int(has_parking),
            'year_built': year_built,
            'distance_to_metro': distance_to_metro,
            'schools_nearby': schools_nearby,
        }
        
        # Add engineered features
        input_data['property_age'] = 2024 - year_built
        input_data['amenity_score'] = int(has_elevator) + int(has_balcony) + int(has_parking)
        input_data['room_density'] = area_sqm / (bedrooms + bathrooms)
        input_data['metro_accessibility'] = 1 / (distance_to_metro + 0.1)
        
        # One-hot encode categorical variables
        for prop_type in ['apartment', 'house', 'studio']:
            input_data[f'property_type_{prop_type}'] = int(property_type == prop_type)
        
        for arr in [f"75{i:03d}" for i in range(1, 21)]:
            input_data[f'arrondissement_{arr}'] = int(arrondissement == arr)
        
        # Create DataFrame with correct column order
        input_df = pd.DataFrame([input_data])
        
        # Ensure all required columns are present
        for col in feature_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        
        # Reorder columns to match training data
        input_df = input_df[feature_columns]
        
        # Apply scaling if needed
        if scaler is not None:
            input_scaled = scaler.transform(input_df)
            input_for_prediction = pd.DataFrame(input_scaled, columns=feature_columns)
        else:
            input_for_prediction = input_df
        
        # Make prediction
        prediction = model.predict(input_for_prediction)[0]
        
        # Display results
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                label="🏷️ Predicted Price", 
                value=f"€{prediction:,.0f}",
                help="Estimated market price for this property"
            )
            
            price_per_sqm = prediction / area_sqm
            st.metric(
                label="📊 Price per m²", 
                value=f"€{price_per_sqm:,.0f}",
                help="Price per square meter"
            )
        
        with col2:
            # Create a simple visualization
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prediction,
                title = {'text': "Predicted Price (€)"},
                gauge = {
                    'axis': {'range': [None, prediction * 1.5]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, prediction * 0.7], 'color': "lightgray"},
                        {'range': [prediction * 0.7, prediction * 1.2], 'color': "gray"}],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': prediction}
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Property summary
        st.subheader("📋 Property Summary")
        summary_data = {
            "Feature": ["Property Type", "Location", "Area", "Bedrooms", "Bathrooms", 
                       "Floor", "Year Built", "Distance to Metro", "Amenities"],
            "Value": [property_type.title(), arrondissement, f"{area_sqm} m²", 
                     bedrooms, bathrooms, floor, year_built, 
                     f"{distance_to_metro} km", f"{input_data['amenity_score']}/3"]
        }
        
        summary_df = pd.DataFrame(summary_data)
        st.table(summary_df)
    
    # Model information
    st.sidebar.markdown("---")
    st.sidebar.info(
        "🤖 This predictor uses advanced machine learning algorithms "
        "trained on Paris housing data to estimate property prices."
    )

if __name__ == "__main__":
    main()
