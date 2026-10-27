import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_synthetic_data(days=365):
    """Generates realistic synthetic hourly electricity demand data."""
    np.random.seed(42)
    start_date = datetime(2025, 1, 1, 0, 0, 0)
    total_hours = days * 24
    timestamps = [start_date + timedelta(hours=i) for i in range(total_hours)]

    hours = np.array([ts.hour for ts in timestamps])
    days_of_week = np.array([ts.weekday() for ts in timestamps])
    day_of_year = np.array([ts.timetuple().tm_yday for ts in timestamps])

    base = 1500
    daily_pattern = 300 * np.sin(2 * np.pi * (hours - 6) / 24)
    weekly_pattern = -150 * (days_of_week >= 5)
    seasonal_pattern = 400 * np.cos(2 * np.pi * (day_of_year - 15) / 365)
    noise = np.random.normal(0, 50, total_hours)

    demand = base + daily_pattern + weekly_pattern + seasonal_pattern + noise
    demand = np.maximum(demand, 200)

    df = pd.DataFrame({
        'timestamp': timestamps,
        'demand': demand
    })
    return df


def load_and_preprocess_data(file_obj_or_path, timestamp_col='timestamp', demand_col='demand'):
    """Loads CSV data, parses timestamps, cleans missing values, and validates columns."""
    try:
        if isinstance(file_obj_or_path, pd.DataFrame):
            df = file_obj_or_path.copy()
        else:
            df = pd.read_csv(file_obj_or_path)

        if timestamp_col not in df.columns or demand_col not in df.columns:
            raise ValueError(f"Columns '{timestamp_col}' and '{demand_col}' must exist in the dataset.")

        df = df[[timestamp_col, demand_col]].rename(columns={timestamp_col: 'timestamp', demand_col: 'demand'})

        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])

        df['demand'] = pd.to_numeric(df['demand'], errors='coerce')
        df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
        df['demand'] = df['demand'].interpolate(method='linear').ffill().bfill()

        return df
    except Exception as e:
        raise RuntimeError(f"Error processing dataset: {str(e)}")
