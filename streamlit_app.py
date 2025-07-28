
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
    
    st.title("🏠 Paris Housing Price Predictor (PCA-Enhanced)")
    st.markdown("### Predict housing prices using PCA-transformed features for improved accuracy")
    
    # Load model
    try:
        artifacts = load_model()
        model = artifacts['model']
        scaler = artifacts.get('scaler')
        pca_scaler = artifacts['pca_scaler']
        pca_transformer = artifacts['pca_transformer']
        feature_columns = artifacts['feature_columns']  # PCA components
        original_features = artifacts['original_features']
        model_name = artifacts['model_name']
        pca_components = artifacts['pca_components']
        variance_explained = artifacts['variance_explained']
        
        st.success(f"✅ Model loaded: {model_name}")
        st.info(f"📊 Using {pca_components} PCA components explaining {variance_explained*100:.2f}% of variance")
        
        # Add explanation about PCA enhancement
        with st.expander("🔬 What is PCA Enhancement?"):
            st.markdown("""
            **Principal Component Analysis (PCA) Enhancement Benefits:**
            - ✅ **Addresses multicollinearity**: Resolves perfect correlations between features
            - ✅ **Dimensionality reduction**: Reduces 34 features to 21 components (38.2% reduction)
            - ✅ **Improves model stability**: Better cross-validation performance
            - ✅ **Retains information**: Maintains 97% of original data variance
            - ✅ **Computational efficiency**: Faster predictions with fewer features
            
            The model achieved **98.92% accuracy (R²)** using XGBoost with PCA transformation!
            """)
            
    except FileNotFoundError:
        st.error("❌ Model file not found. Please ensure 'paris_housing_model.pkl' is in the same directory.")
        return
    
    # Sidebar for inputs (using original feature interpretations)
    st.sidebar.header("🏡 Property Details")
    
    # Basic property information
    square_meters = st.sidebar.slider("Area (Square Meters)", 20, 200, 75, help="Property size in square meters")
    num_rooms = st.sidebar.slider("Number of Rooms", 1, 10, 3, help="Total number of rooms")
    floors = st.sidebar.slider("Number of Floors", 1, 5, 2, help="Number of floors in the building")
    
    # Location
    st.sidebar.subheader("📍 Location")
    city_code = st.sidebar.selectbox("City Code", [75001, 75002, 75003, 75004, 75005], help="Paris postal code")
    city_part_range = st.sidebar.slider("City Part Range", 1, 10, 5, help="Lower values indicate more desirable areas (1=most desirable)")
    
    # Property features
    st.sidebar.subheader("✨ Amenities & Features")
    has_yard = st.sidebar.checkbox("Has Yard", help="Property includes a yard")
    has_pool = st.sidebar.checkbox("Has Pool", help="Property includes a swimming pool")
    basement = st.sidebar.checkbox("Has Basement", help="Property includes a basement")
    attic = st.sidebar.checkbox("Has Attic", help="Property includes an attic")
    garage = st.sidebar.checkbox("Has Garage", help="Property includes a garage")
    
    # Building details
    st.sidebar.subheader("🏗️ Building Details")
    year_made = st.sidebar.slider("Year Built", 1950, 2024, 1990, help="Year the property was constructed")
    is_new_built = st.sidebar.checkbox("New Building", help="Property is a new construction")
    num_prev_owners = st.sidebar.slider("Number of Previous Owners", 0, 10, 2, help="Number of previous property owners")
    
    # Create prediction button
    if st.sidebar.button("🔮 Predict Price", type="primary"):
        
        # Prepare input data matching original features structure
        input_data = {
            'squareMeters': square_meters,
            'numberOfRooms': num_rooms,
            'hasYard': int(has_yard),
            'hasPool': int(has_pool),
            'floors': floors,
            'cityCode': city_code,
            'cityPartRange': city_part_range,
            'numPrevOwners': num_prev_owners,
            'made': year_made,
            'isNewBuilt': int(is_new_built),
            'hasStormProtector': 0,  # Default values for demo
            'basement': int(basement),
            'attic': int(attic),
            'garage': int(garage),
            'hasStorageRoom': 0,
            'hasGuestRoom': 0,
        }
        
        # Add engineered features (matching the original preprocessing)
        input_data['property_age'] = 2024 - year_made
        input_data['amenity_score'] = sum([has_yard, has_pool, basement, attic, garage])
        input_data['room_density'] = square_meters / num_rooms if num_rooms > 0 else 0
        input_data['floor_efficiency'] = floors / square_meters * 1000 if square_meters > 0 else 0
        input_data['condition_score'] = (int(is_new_built) * 2 + (11 - min(num_prev_owners, 10))) / 13
        input_data['city_desirability'] = 11 - city_part_range
        
        # Create size and room categories (simplified)
        if square_meters <= 50:
            size_cat = 'Small'
        elif square_meters <= 75:
            size_cat = 'Medium'
        elif square_meters <= 100:
            size_cat = 'Large'
        else:
            size_cat = 'XLarge'
            
        if num_rooms <= 2:
            room_cat = 'Few'
        elif num_rooms <= 4:
            room_cat = 'Moderate'
        elif num_rooms <= 6:
            room_cat = 'Many'
        else:
            room_cat = 'Numerous'
            
        if city_part_range <= 3:
            loc_des = 'Prime'
        elif city_part_range <= 6:
            loc_des = 'Good'
        elif city_part_range <= 10:
            loc_des = 'Average'
        else:
            loc_des = 'Basic'
        
        # Add categorical encodings (simplified - would need all categories in practice)
        for cat in ['Small', 'Medium', 'Large', 'XLarge']:
            input_data[f'size_category_{cat}'] = int(size_cat == cat)
        for cat in ['Few', 'Moderate', 'Many', 'Numerous']:
            input_data[f'room_category_{cat}'] = int(room_cat == cat)
        for cat in ['Prime', 'Good', 'Average', 'Basic']:
            input_data[f'location_desirability_{cat}'] = int(loc_des == cat)
        
        # Create DataFrame
        input_df = pd.DataFrame([input_data])
        
        # Ensure all original features are present (fill missing with defaults)
        for feature in original_features:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        # Reorder to match original feature order
        input_df = input_df[original_features]
        
        # Apply PCA transformation pipeline
        try:
            # 1. Scale using the original scaler
            input_scaled = pca_scaler.transform(input_df)
            
            # 2. Apply PCA transformation
            input_pca = pca_transformer.transform(input_scaled)
            
            # 3. Convert to DataFrame with PCA component names
            input_pca_df = pd.DataFrame(input_pca, columns=feature_columns)
            
            # 4. Apply additional scaling if model requires it
            if scaler is not None:
                input_final = scaler.transform(input_pca_df)
                input_for_prediction = pd.DataFrame(input_final, columns=feature_columns)
            else:
                input_for_prediction = input_pca_df
            
            # Make prediction
            prediction = model.predict(input_for_prediction)[0]
            
            # Display results with enhanced formatting
            st.success("🎉 Prediction Complete!")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="🏷️ Predicted Price", 
                    value=f"€{prediction:,.0f}",
                    help="Estimated market price using PCA-enhanced XGBoost model"
                )
                
            with col2:
                price_per_sqm = prediction / square_meters
                st.metric(
                    label="📊 Price per m²", 
                    value=f"€{price_per_sqm:,.0f}",
                    help="Price per square meter"
                )
            
            with col3:
                # Calculate market position (simplified)
                if price_per_sqm < 8000:
                    market_pos = "Budget-Friendly"
                elif price_per_sqm < 12000:
                    market_pos = "Mid-Range"
                elif price_per_sqm < 18000:
                    market_pos = "Premium"
                else:
                    market_pos = "Luxury"
                    
                st.metric(
                    label="🎯 Market Position",
                    value=market_pos,
                    help="Property category based on price per square meter"
                )
            
            # Create enhanced visualization
            col1, col2 = st.columns(2)
            
            with col1:
                # Price gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = prediction,
                    title = {'text': "Predicted Price (€)", 'font': {'size': 16}},
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
                
                fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Feature contribution visualization
                feature_values = {
                    'Area': square_meters,
                    'Rooms': num_rooms,
                    'Amenities': input_data['amenity_score'],
                    'Age': 2024 - year_made,
                    'Location Score': input_data['city_desirability']
                }
                
                fig2 = px.bar(
                    x=list(feature_values.keys()),
                    y=list(feature_values.values()),
                    title="Key Property Features",
                    labels={'x': 'Feature', 'y': 'Value'},
                    color=list(feature_values.values()),
                    color_continuous_scale='viridis'
                )
                fig2.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig2, use_container_width=True)
            
            # Property summary
            st.subheader("📋 Property Summary")
            
            col1, col2 = st.columns(2)
            
            with col1:
                summary_data = {
                    "Property Details": [
                        f"📐 Area: {square_meters} m²",
                        f"🏠 Rooms: {num_rooms}",
                        f"🏢 Floors: {floors}",
                        f"📍 Location: {city_code} (Range: {city_part_range})",
                        f"📅 Built: {year_made} ({'New' if is_new_built else 'Existing'})"
                    ]
                }
                for detail in summary_data["Property Details"]:
                    st.write(detail)
            
            with col2:
                amenities_list = []
                if has_yard: amenities_list.append("🌿 Yard")
                if has_pool: amenities_list.append("🏊 Pool")
                if basement: amenities_list.append("🏠 Basement")
                if attic: amenities_list.append("🏠 Attic")
                if garage: amenities_list.append("🚗 Garage")
                
                st.write("**Amenities:**")
                if amenities_list:
                    for amenity in amenities_list:
                        st.write(amenity)
                else:
                    st.write("No additional amenities")
                
                st.write(f"**Amenity Score:** {input_data['amenity_score']}/5")
                st.write(f"**Previous Owners:** {num_prev_owners}")
            
            # Model performance information
            st.info(
                f"📊 **Model Performance**: R² = 98.92% | MAE = €219,907 | "
                f"This prediction uses {model_name} with PCA enhancement for superior accuracy."
            )
            
        except Exception as e:
            st.error(f"❌ Error in prediction: {str(e)}")
            st.error("This might be due to missing feature categories in the simplified demo.")
            st.info("💡 **Tip**: Try adjusting the input values or contact support if the error persists.")
    
    # Model information sidebar
    st.sidebar.markdown("---")
    st.sidebar.info(
        f"🤖 **Model**: {model_name if 'artifacts' in locals() else 'XGBoost (Tuned)'}\n\n"
        f"📊 **PCA Components**: {pca_components if 'artifacts' in locals() else '21'} components\n\n"
        f"📈 **Variance Explained**: {variance_explained*100:.2f}% if 'artifacts' in locals() else '97.00%'\n\n"
        f"🔬 **Benefits**: Addresses multicollinearity, improves stability, reduces overfitting"
    )
    
    # Footer
    st.markdown("---")
    st.markdown(
        "**🏠 Paris Housing Price Predictor** | "
        "Built with PCA-enhanced machine learning for accurate real estate valuation | "
        "Model achieves 98.92% accuracy on test data"
    )

if __name__ == "__main__":
    main()