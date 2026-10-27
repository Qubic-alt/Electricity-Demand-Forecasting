import pandas as pd
import numpy as np


def create_features(df):
    """Extracts temporal, lag, and rolling features for time-series forecasting."""
    df = df.copy()

    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    df['lag_1h'] = df['demand'].shift(1)
    df['lag_24h'] = df['demand'].shift(24)
    df['lag_168h'] = df['demand'].shift(168)

    df['rolling_mean_24h'] = df['demand'].shift(1).rolling(window=24).mean()
    df['rolling_std_24h'] = df['demand'].shift(1).rolling(window=24).std()

    df = df.dropna().reset_index(drop=True)
    return df


def get_feature_columns():
    return [
        'hour', 'day_of_week', 'month', 'day_of_year', 'is_weekend',
        'lag_1h', 'lag_24h', 'lag_168h', 'rolling_mean_24h', 'rolling_std_24h'
    ]
