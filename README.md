# Paris Housing Price Prediction - ML Project

## 🏠 Overview

This comprehensive machine learning project predicts housing prices in Paris using multiple regression models. The project demonstrates a complete ML pipeline from data preprocessing to model deployment.

## 🎯 Project Objectives

- Develop and compare multiple regression models (Linear Regression, Random Forest, XGBoost)
- Perform comprehensive data preprocessing and feature engineering
- Evaluate models using MAE, RMSE, and R-squared with k-fold cross-validation
- Analyze feature importance using model-specific methods and SHAP values
- Deploy an interactive web application for real-time price predictions

## 📁 Project Structure

```
├── Paris_Housing_Price_Prediction.ipynb    # Main Jupyter notebook
├── streamlit_app.py                         # Web application
├── requirements.txt                         # Python dependencies
├── README.md                               # This file
└── paris_housing_model.pkl                # Saved model artifacts (generated)
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone or download the project files
2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Jupyter Notebook

1. Open Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `Paris_Housing_Price_Prediction.ipynb`

3. Run all cells sequentially

### Running the Web Application

1. After running the notebook (which saves the model), launch the Streamlit app:
```bash
streamlit run streamlit_app.py
```

2. Open your web browser and navigate to the provided local URL (typically `http://localhost:8501`)

## 📊 Model Performance

The project compares three regression models:

| Model | Test MAE | Test RMSE | Test R² | Cross-Val R² |
|-------|----------|-----------|---------|--------------|
| Linear Regression | €1,510 | €1,922 | 1.0000 | 1.0000 |
| Random Forest | €3,165 | €3,979 | 1.0000 | 1.0000 |
| XGBoost | €10,232 | €12,098 | 1.0000 | 1.0000 |

**Best Model:** Linear Regression (selected for deployment)

## 🔍 Key Features

### Data Preprocessing
- Missing value imputation
- One-hot encoding for categorical variables
- Feature engineering (property age, amenity scores, room density, etc.)
- Feature scaling and normalization

### Model Development
- Linear Regression with feature scaling
- Random Forest with hyperparameter tuning
- XGBoost with grid search optimization
- K-fold cross-validation (k=5)

### Feature Importance Analysis
- Model-specific feature importance
- SHAP (SHapley Additive exPlanations) values
- Correlation analysis and visualization

### Web Application Features
- Interactive property input form
- Real-time price predictions
- Price per square meter calculations
- Property summary and market insights
- Model performance metrics display

## 💡 Key Insights

### Most Important Features:
1. **Square Meters** - Primary driver of property value
2. **Room Density** - Efficiency of space utilization  
3. **Floor Efficiency** - Building height optimization
4. **Amenity Score** - Combined amenities value
5. **Location Factors** - City part and accessibility

### Real-World Applications:
- **Real Estate Professionals:** Quick property valuation and market analysis
- **Property Buyers/Sellers:** Fair price assessment and negotiation support
- **Property Developers:** Site selection and feature prioritization

## 🛠️ Technical Implementation

### Libraries Used:
- **Data Processing:** pandas, numpy
- **Machine Learning:** scikit-learn, xgboost
- **Visualization:** matplotlib, seaborn, plotly
- **Model Interpretation:** shap
- **Web Application:** streamlit
- **Model Persistence:** joblib

### Model Evaluation Metrics:
- **MAE (Mean Absolute Error):** Average prediction error in euros
- **RMSE (Root Mean Square Error):** Penalizes larger errors more heavily
- **R² Score:** Coefficient of determination (proportion of variance explained)
- **Cross-Validation:** 5-fold CV for robust performance estimation

## 📈 Future Improvements

1. **Data Enhancement:**
   - Incorporate real market data from APIs
   - Add external factors (economic indicators, neighborhood development)
   - Include property condition and renovation status

2. **Model Improvements:**
   - Ensemble methods combining multiple models
   - Deep learning approaches for complex patterns
   - Time series analysis for price trends

3. **Deployment Enhancements:**
   - Cloud deployment (Heroku, AWS, Google Cloud)
   - Real-time data integration
   - Mobile-responsive design
   - User feedback collection system

## 🔧 Troubleshooting

### Common Issues:

1. **Missing Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Dataset Not Found:**
   - The notebook creates a synthetic dataset if the original file is not found
   - Update the data path in the notebook if using your own dataset

3. **Model File Not Found:**
   - Run the Jupyter notebook first to generate the model file
   - The Streamlit app includes a demo mode if the model file is missing

## 📄 License

This project is for educational and demonstration purposes. Feel free to use and modify as needed.

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- Additional feature engineering techniques
- More sophisticated model architectures
- Enhanced visualization capabilities
- Performance optimizations

## 📞 Support

For questions or issues, please review the notebook documentation and code comments. The project is designed to be self-contained and educational.

---

*This project demonstrates a complete machine learning pipeline for real estate price prediction, showcasing best practices in data science and model deployment.*
