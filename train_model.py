import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

print("--- Loading Dataset ---")
df = pd.read_csv("auto-mpg.csv")
df['horsepower'] = pd.to_numeric(df['horsepower'].replace('?', np.nan), errors='coerce')

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

print("\n--- Model Evaluation Results ---")
for name, model in models.items():
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('regressor', model)
    ])
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    print(f"[{name}] R2: {r2:.4f} | RMSE: {rmse:.4f} | MAE: {mae:.4f}")
