# Electricity Demand Forecasting Using Machine Learning

A complete, production-ready machine learning system for predicting electrical load demand with an interactive Streamlit dashboard.

## Features
- Automated synthetic dataset generator & CSV uploader
- Chronological time-series preprocessing & feature engineering
- Multiple ML models (Persistence Baseline, Linear Regression, Decision Tree, Random Forest, Gradient Boosting)
- Interactive multi-step forecasting (1 to 48 hours)
- Comprehensive evaluation metrics (MAE, RMSE, R2, MAPE)
- Visualizations with Plotly

## Run the dashboard

The dashboard uses a dark neon theme and supports both account registration and
sign-in. Registered accounts are saved locally in `users.db`; passwords are
stored as salted hashes, not plaintext. The database is excluded from source
control.

In PowerShell:

```powershell
streamlit run app.py
```

Use **Create Account** to register, then **Sign In** to open the dashboard.
Optionally set `POWERFORECAST_USER_ID` and `POWERFORECAST_PASSWORD` in the
environment to enable an additional environment-configured account. After
signing in, use **Log out** in the sidebar to return to the login page.