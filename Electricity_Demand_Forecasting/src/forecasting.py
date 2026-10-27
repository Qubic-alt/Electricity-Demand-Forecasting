import pandas as pd
from src.feature_engineering import get_feature_columns


def recursive_forecast(model, last_known_df, horizon=24):
    """Performs recursive multi-step forecasting for future horizons."""
    forecasts = []
    current_df = last_known_df.copy()

    feature_cols = get_feature_columns()

    for step in range(horizon):
        last_row = current_df.iloc[-1]
        next_timestamp = last_row['timestamp'] + pd.Timedelta(hours=1)

        hour = next_timestamp.hour
        day_of_week = next_timestamp.dayofweek
        month = next_timestamp.month
        day_of_year = next_timestamp.dayofyear
        is_weekend = 1 if day_of_week >= 5 else 0

        lag_1h = current_df['demand'].iloc[-1]
        lag_24h = current_df['demand'].iloc[-24] if len(current_df) >= 24 else lag_1h
        lag_168h = current_df['demand'].iloc[-168] if len(current_df) >= 168 else lag_1h

        recent_24 = current_df['demand'].tail(24)
        rolling_mean_24h = recent_24.mean()
        rolling_std_24h = recent_24.std() if len(recent_24) > 1 else 0.0

        X_next = pd.DataFrame([{
            'hour': hour,
            'day_of_week': day_of_week,
            'month': month,
            'day_of_year': day_of_year,
            'is_weekend': is_weekend,
            'lag_1h': lag_1h,
            'lag_24h': lag_24h,
            'lag_168h': lag_168h,
            'rolling_mean_24h': rolling_mean_24h,
            'rolling_std_24h': rolling_std_24h
        }])

        pred_demand = model.predict(X_next)[0]
        forecasts.append({'timestamp': next_timestamp, 'forecast_demand': pred_demand})

        new_row = pd.DataFrame({'timestamp': [next_timestamp], 'demand': [pred_demand]})
        current_df = pd.concat([current_df, new_row], ignore_index=True)

    return pd.DataFrame(forecasts)
