import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib
import os

class BaselineModel:
    """Persistence baseline model predicting the demand from 24 hours ago."""
    def fit(self, X, y):
        pass
    def predict(self, X):
        # Assumes lag_24h is present in X
        if 'lag_24h' in X.columns:
            return X['lag_24h'].values
        return np.zeros(len(X))

def train_models(X_train, y_train):
    """Trains multiple machine learning models and returns a dictionary of fitted models."""
    models = {
        'Baseline (24h Lag)': BaselineModel(),
        'Linear Regression': LinearRegression(),
        'Decision Tree': DecisionTreeRegressor(random_state=42),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    fitted_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model
        
    return fitted_models

def save_model(model, feature_cols, model_name, path='models/best_model.joblib'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        'model': model,
        'feature_cols': feature_cols,
        'model_name': model_name
    }
    joblib.dump(payload, path)

def load_saved_model(path='models/best_model.joblib'):
    if os.path.exists(path):
        return joblib.load(path)
    return None