# Automobile Fuel Efficiency (MPG) ML Prediction Project

## Application Features
1. **Typing Box Inputs**: Direct numerical typing input fields for Displacement, Horsepower, Vehicle Weight, Acceleration (0-60mph), and Model Year.
2. **Animated Visuals**: Dynamic animated scatter plots showing MPG vs Weight trend color-coded by Horsepower.
3. **Automated ML Preprocessing**: Pipeline handling missing horsepower imputation, Standard Scaling, and One-Hot Encoding.
4. **Multi-Model Evaluation**: Ridge Linear Regression, Random Forest Regressor, and k-NN Regressor with R², RMSE, and MAE benchmarks.

## Included Files
- `app.py`: Complete Streamlit web application.
- `auto-mpg.csv`: Cleaned UCI Auto-MPG dataset.
- `train_model.py`: Standalone ML model evaluation script.
- `requirements.txt`: Python dependencies.
- `README.md`: Documentation.

## Running the Application
```bash
pip install -r requirements.txt
streamlit run app.py
```
