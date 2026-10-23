"""
Streamlit Application for Automobile Fuel Efficiency (MPG) Prediction
Includes Typing Boxes, Preprocessing Pipeline, 3 ML Models, Evaluation Matrix, and Animated MPG Visualizations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

st.set_page_config(
    page_title="Auto MPG Predictor & Animated EDA",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Automobile Fuel Efficiency (MPG) Machine Learning Predictor")
st.markdown("Predict vehicle fuel efficiency using technical engine specs, preprocessing pipelines, and machine learning models.")

@st.cache_data
def load_data():
    df = pd.read_csv("auto-mpg.csv")
    df['horsepower'] = pd.to_numeric(df['horsepower'].replace('?', np.nan), errors='coerce')
    return df

df = load_data()

# Data Preprocessing Pipeline setup
X = df.drop(columns=['mpg', 'car_name'])
y = df['mpg']

num_features = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year']
cat_features = ['origin']

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_features),
    ('cat', OneHotEncoder(drop='first', sparse_output=False), cat_features)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'Ridge Linear Regression': Ridge(alpha=1.0),
    'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42),
    'k-Nearest Neighbors (k-NN)': KNeighborsRegressor(n_neighbors=5)
}

trained_models = {}
eval_results = []

for name, model in models.items():
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    pipe.fit(X_train, y_train)
    trained_models[name] = pipe
    
    y_pred = pipe.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    eval_results.append({'Model': name, 'R2 Score': r2, 'RMSE': rmse, 'MAE': mae})

eval_df = pd.DataFrame(eval_results).sort_values(by='R2 Score', ascending=False)

tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Real-Time Prediction", 
    "🎬 Animated Visuals", 
    "📊 Model Comparison", 
    "📁 Dataset Overview"
])

with tab1:
    st.subheader("Input Vehicle Specifications (Typing Boxes)")
    col1, col2 = st.columns(2)
    
    with col1:
        cylinders = st.selectbox("Engine Cylinders", options=[3, 4, 5, 6, 8], index=1)
        displacement = st.number_input("Displacement (cu. in.)", min_value=60.0, max_value=500.0, value=193.0, step=1.0)
        horsepower = st.number_input("Horsepower (hp)", min_value=40.0, max_value=300.0, value=100.0, step=1.0)
        weight = st.number_input("Vehicle Weight (lbs)", min_value=1500.0, max_value=6000.0, value=2970.0, step=10.0)
        
    with col2:
        acceleration = st.number_input("0-60 mph Acceleration (sec)", min_value=5.0, max_value=30.0, value=15.5, step=0.1)
        model_year = st.number_input("Model Year (e.g. 70 to 82)", min_value=70, max_value=82, value=76, step=1)
        origin_str = st.selectbox("Country of Origin", options=["USA (1)", "Europe (2)", "Japan (3)"])
        origin = 1 if "USA" in origin_str else (2 if "Europe" in origin_str else 3)
        selected_model_name = st.selectbox("Select ML Model Algorithm", list(models.keys()))

    input_df = pd.DataFrame([{
        'cylinders': cylinders,
        'displacement': displacement,
        'horsepower': horsepower,
        'weight': weight,
        'acceleration': acceleration,
        'model_year': model_year,
        'origin': origin
    }])
    
    if st.button("🚀 Calculate Predicted MPG", type="primary"):
        chosen_pipeline = trained_models[selected_model_name]
        pred_mpg = chosen_pipeline.predict(input_df)[0]
        
        st.success(f"### Predicted Fuel Efficiency: **{pred_mpg:.2f} MPG**")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Est. CO2 Emissions", f"{round(8887 / pred_mpg)} g/mi")
        c2.metric("Est. Annual Fuel Cost", f"${round((15000 / pred_mpg) * 3.50):,}")
        c3.metric("EPA Rating", "High Efficiency" if pred_mpg >= 28 else ("Moderate" if pred_mpg >= 19 else "Low Efficiency"))

with tab2:
    st.subheader("🎬 Animated Scatter Trend: Vehicle Weight vs. MPG")
    st.write("Click 'Play Animation' to stream data points dynamically onto the chart.")
    
    if st.button("▶️ Play Animated Scatter Plot"):
        clean_df = df.dropna(subset=['horsepower']).sort_values('weight').reset_index(drop=True)
        chart_holder = st.empty()
        
        for i in range(5, len(clean_df) + 1, 2):
            sub_df = clean_df.iloc[:i]
            chart_holder.scatter_chart(
                data=sub_df,
                x='weight',
                y='mpg',
                color='horsepower',
                size=50
            )
            time.sleep(0.05)
        st.success("Animation completed!")
    else:
        st.scatter_chart(
            data=df.dropna(subset=['horsepower']),
            x='weight',
            y='mpg',
            color='horsepower'
        )

with tab3:
    st.subheader("Model Benchmark & Evaluation Matrix")
    st.dataframe(eval_df.style.highlight_max(subset=['R2 Score'], color='#d1e7dd'))
    
    st.subheader("Feature Importance (Random Forest)")
    rf_model = trained_models['Random Forest Regressor'].named_steps['regressor']
    encoded_cats = ['origin_2', 'origin_3']
    feature_names = num_features + encoded_cats
    
    importances = pd.DataFrame({
        'Feature': feature_names,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    st.bar_chart(importances.set_index('Feature'))

with tab4:
    st.subheader("Auto-MPG Dataset")
    st.dataframe(df)
