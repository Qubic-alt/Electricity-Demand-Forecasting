import streamlit as st
import pandas as pd
import numpy as np
import os
from src.auth import credentials_match, register_user, verify_user
from src.data_processing import generate_synthetic_data, load_and_preprocess_data
from src.feature_engineering import create_features, get_feature_columns
from src.model_training import train_models, save_model, load_saved_model
from src.forecasting import recursive_forecast
from src.evaluation import evaluate_predictions
from src.visualization import plot_historical_demand, plot_actual_vs_predicted, plot_feature_importance, plot_residuals

st.set_page_config(page_title="Electricity Demand Forecasting", page_icon="⚡", layout="wide")

st.markdown("""
    <style>
    :root {
        color-scheme: dark;
        --neon-cyan: #00f5ff;
        --neon-green: #39ff88;
        --neon-purple: #bd5cff;
        --neon-pink: #ff4fd8;
        --surface: #0d1220;
        --surface-raised: #121a2b;
        --text-primary: #e8f7ff;
        --text-muted: #8ca1b8;
    }
    .stApp {
        background:
            radial-gradient(ellipse at 12% 0%, rgba(0, 245, 255, 0.09), transparent 35%),
            radial-gradient(ellipse at 90% 12%, rgba(189, 92, 255, 0.10), transparent 30%),
            #070a12;
        color: var(--text-primary);
    }
    [data-testid="stHeader"] { background: rgba(7, 10, 18, 0.75); }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1020 0%, #090c16 100%);
        border-right: 1px solid rgba(0, 245, 255, 0.18);
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2 {
        color: var(--neon-cyan);
        text-shadow: 0 0 18px rgba(0, 245, 255, 0.35);
    }
    h1, h2, h3 { color: var(--text-primary); }
    h1 { text-shadow: 0 0 24px rgba(0, 245, 255, 0.2); }
    p, label, [data-testid="stMarkdownContainer"] { color: var(--text-primary); }
    [data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(18, 26, 43, 0.96), rgba(11, 16, 29, 0.96));
        border: 1px solid rgba(0, 245, 255, 0.25);
        border-radius: 14px;
        padding: 18px;
        box-shadow: 0 0 22px rgba(0, 245, 255, 0.08);
    }
    [data-testid="stMetricLabel"] { color: var(--text-muted); }
    [data-testid="stMetricValue"] { color: var(--neon-cyan); }
    .stButton > button, [data-testid="stFormSubmitButton"] > button,
    .stDownloadButton > button {
        color: #061014;
        background: linear-gradient(100deg, var(--neon-cyan), var(--neon-green));
        border: 0;
        border-radius: 9px;
        font-weight: 700;
        box-shadow: 0 0 18px rgba(0, 245, 255, 0.23);
        transition: transform 150ms ease, box-shadow 150ms ease;
    }
    .stButton > button:hover, [data-testid="stFormSubmitButton"] > button:hover,
    .stDownloadButton > button:hover {
        color: #061014;
        transform: translateY(-1px);
        box-shadow: 0 0 24px rgba(0, 245, 255, 0.42);
        border: 0;
    }
    [data-testid="stTextInput"] input, [data-testid="stSelectbox"] > div,
    [data-testid="stNumberInput"] input {
        background-color: var(--surface-raised);
        color: var(--text-primary);
        border-color: rgba(0, 245, 255, 0.3);
        border-radius: 8px;
    }
    [data-testid="stDataFrame"], [data-testid="stTable"] {
        border: 1px solid rgba(189, 92, 255, 0.25);
        border-radius: 10px;
    }
    button[data-baseweb="tab"] { color: var(--text-muted); }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: var(--neon-cyan);
        border-bottom-color: var(--neon-cyan);
    }
    [data-testid="stPlotlyChart"] {
        border: 1px solid rgba(0, 245, 255, 0.19);
        border-radius: 12px;
        background: rgba(13, 18, 32, 0.65);
        padding: 8px;
    }
    hr { border-color: rgba(0, 245, 255, 0.2); }
    </style>
""", unsafe_allow_html=True)

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

configured_user_id = os.getenv("POWERFORECAST_USER_ID")
configured_password = os.getenv("POWERFORECAST_PASSWORD")

if not st.session_state.authenticated:
    _, login_column, _ = st.columns([1, 1.15, 1])
    with login_column:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown("## ⚡ PowerForecast AI")
        st.markdown("### GRID INTELLIGENCE · SECURE ACCESS")
        st.markdown("Sign in or create an account to explore demand analytics and forecasts.")
        sign_in_tab, registration_tab = st.tabs(["Sign In", "Create Account"])

        with sign_in_tab:
            with st.form("login_form"):
                user_id = st.text_input(
                    "User ID",
                    placeholder="Enter your user ID",
                    key="signin_user_id",
                )
                password = st.text_input(
                    "Login password",
                    type="password",
                    placeholder="Enter your password",
                    key="signin_password",
                )
                submitted = st.form_submit_button(
                    "SIGN IN",
                    use_container_width=True,
                )
            if submitted:
                registered_user = verify_user(user_id, password)
                configured_user = credentials_match(
                    user_id,
                    password,
                    configured_user_id or "",
                    configured_password or "",
                )
                if registered_user or configured_user:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("The user ID or password is incorrect.")

        with registration_tab:
            with st.form("registration_form"):
                new_user_id = st.text_input(
                    "Choose a user ID",
                    placeholder="Enter a user ID",
                    key="register_user_id",
                )
                new_password = st.text_input(
                    "Create a password",
                    type="password",
                    placeholder="Use at least 8 characters",
                    key="register_password",
                )
                confirm_password = st.text_input(
                    "Confirm password",
                    type="password",
                    placeholder="Enter your password again",
                    key="confirm_password",
                )
                register_submitted = st.form_submit_button(
                    "CREATE ACCOUNT",
                    use_container_width=True,
                )
            if register_submitted:
                if new_password != confirm_password:
                    st.error("The passwords do not match.")
                else:
                    try:
                        registered = register_user(new_user_id, new_password)
                    except ValueError as error:
                        st.error(str(error))
                    else:
                        if registered:
                            st.success("Account created. Sign in with your new credentials.")
                        else:
                            st.error("That user ID is already registered.")
    st.stop()

# Sidebar Navigation
st.sidebar.title("⚡ PowerForecast AI")
page = st.sidebar.radio("Navigation", ["Home Dashboard", "Data Explorer", "Demand Forecast", "Model Training", "Model Insights"])
if st.sidebar.button("Log out", use_container_width=True):
    st.session_state.authenticated = False
    st.rerun()

# Session State Initialization
if 'df' not in st.session_state:
    st.session_state.df = generate_synthetic_data(days=180)
if 'models' not in st.session_state:
    st.session_state.models = None
if 'best_model_name' not in st.session_state:
    st.session_state.best_model_name = None

df = st.session_state.df

if page == "Home Dashboard":
    st.title("⚡ Electricity Demand Forecasting Dashboard")
    st.markdown("Monitor historical electricity consumption patterns, view statistical summaries, and analyze key grid metrics.")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Observations", f"{len(df):,}")
    with col2:
        st.metric("Average Demand", f"{df['demand'].mean():.2f} kW")
    with col3:
        st.metric("Peak Demand", f"{df['demand'].max():.2f} kW")
    with col4:
        st.metric("Latest Recorded", f"{df['demand'].iloc[-1]:.2f} kW")
        
    st.plotly_chart(plot_historical_demand(df), use_container_width=True)

elif page == "Data Explorer":
    st.title("📂 Data Explorer & CSV Uploader")
    st.markdown("Upload your custom electricity consumption dataset or inspect the active working dataset.")
    
    uploaded_file = st.file_uploader("Upload CSV Dataset", type=['csv'])
    if uploaded_file is not None:
        try:
            df_uploaded = load_and_preprocess_data(uploaded_file)
            st.session_state.df = df_uploaded
            st.success("Dataset successfully uploaded and validated!")
            df = df_uploaded
        except Exception as e:
            st.error(f"Error loading uploaded file: {e}")
            
    if st.button("Load Default Synthetic Dataset"):
        st.session_state.df = generate_synthetic_data(days=180)
        st.success("Default synthetic dataset loaded.")
        df = st.session_state.df

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Dataset Summary Statistics")
        st.dataframe(df.describe(), use_container_width=True)
    with col2:
        st.subheader("Dataset Information")
        st.write(f"**Start Time:** {df['timestamp'].min()}")
        st.write(f"**End Time:** {df['timestamp'].max()}")
        st.write(f"**Missing Values:** {df.isnull().sum().sum()}")
        st.write(f"**Sampling Frequency:** Hourly")

elif page == "Demand Forecast":
    st.title("📈 Future Demand Forecasting")
    st.markdown("Generate recursive multi-step forecasts for upcoming hours using trained machine learning models.")
    
    horizon = st.selectbox("Select Forecasting Horizon", [1, 6, 12, 24, 48], index=3)
    
    # Train/load model if not available
    if st.session_state.models is None:
        st.info("No trained models found in session. Training default models now...")
        df_feat = create_features(df)
        X = df_feat[get_feature_columns()]
        y = df_feat['demand']
        train_size = int(len(X) * 0.8)
        st.session_state.models = train_models(X.iloc[:train_size], y.iloc[:train_size])
        st.session_state.best_model_name = 'Random Forest'
        
    selected_model_name = st.selectbox("Select Model for Forecasting", list(st.session_state.models.keys()), index=3)
    model = st.session_state.models[selected_model_name]
    
    if st.button("Generate Forecast"):
        with st.spinner("Generating recursive forecast..."):
            df_forecast = recursive_forecast(model, df, horizon=horizon)
            st.session_state.df_forecast = df_forecast
            st.success("Forecast generated successfully!")
            
    if 'df_forecast' in st.session_state:
        df_forecast = st.session_state.df_forecast
        
        # Combine recent history and forecast for visualization
        df_recent = df.tail(72)
        fig = plot_actual_vs_predicted(df_recent, df_forecast, title=f"Electricity Demand Forecast ({horizon} Hours Ahead)")
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Forecasted Demand Values")
        st.dataframe(df_forecast, use_container_width=True)
        
        csv = df_forecast.to_csv(index=False).encode('utf-8')
        st.download_button("Download Forecast CSV", data=csv, file_name="electricity_demand_forecast.csv", mime="text/csv")

elif page == "Model Training":
    st.title("🤖 Model Training & Comparison")
    st.markdown("Train multiple machine learning algorithms chronologically and compare performance metrics.")
    
    test_split_pct = st.slider("Test Split Percentage", 10, 40, 20, 5)
    
    if st.button("Run Model Training"):
        with st.spinner("Preprocessing features and training models..."):
            df_feat = create_features(df)
            X = df_feat[get_feature_columns()]
            y = df_feat['demand']
            
            split_idx = int(len(X) * (1 - test_split_pct / 100))
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            timestamp_test = df_feat['timestamp'].iloc[split_idx:]
            
            models = train_models(X_train, y_train)
            st.session_state.models = models
            st.session_state.X_test = X_test
            st.session_state.y_test = y_test
            st.session_state.timestamp_test = timestamp_test
            
            results = []
            for name, m in models.items():
                preds = m.predict(X_test)
                metrics = evaluate_predictions(y_test.values, preds)
                metrics['Model'] = name
                results.append(metrics)
                
            results_df = pd.DataFrame(results).set_index('Model')
            st.session_state.results_df = results_df
            st.success("All models trained and evaluated successfully!")
            
    if 'results_df' in st.session_state:
        st.subheader("Model Performance Comparison")
        st.dataframe(st.session_state.results_df, use_container_width=True)

elif page == "Model Insights":
    st.title("🔍 Model Insights & Error Analysis")
    st.markdown("Analyze feature importances, residuals, and prediction error distributions.")
    
    if st.session_state.models is not None:
        model_name = st.selectbox("Select Model for Insights", list(st.session_state.models.keys()), index=3)
        model = st.session_state.models[model_name]
        
        if hasattr(model, 'feature_importances_'):
            fig_fi = plot_feature_importance(get_feature_columns(), model.feature_importances_)
            st.plotly_chart(fig_fi, use_container_width=True)
        else:
            st.info("Feature importance is not available for this linear or baseline model.")
            
        if 'X_test' in st.session_state and 'y_test' in st.session_state:
            preds = model.predict(st.session_state.X_test)
            fig_res = plot_residuals(st.session_state.y_test.values, preds)
            st.plotly_chart(fig_res, use_container_width=True)
    else:
        st.info("Please train models in the Model Training page first.")
