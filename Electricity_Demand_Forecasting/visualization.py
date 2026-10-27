import plotly.express as px
import plotly.graph_objects as go

NEON_CYAN = "#00f5ff"
NEON_GREEN = "#39ff88"
NEON_PURPLE = "#bd5cff"
NEON_PINK = "#ff4fd8"


def _apply_dark_theme(fig):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(7, 10, 18, 0)",
        plot_bgcolor="rgba(13, 18, 32, 0.92)",
        font=dict(color="#e8f7ff", family="Arial, sans-serif"),
        title_font=dict(color=NEON_CYAN),
        hoverlabel=dict(
            bgcolor="#121a2b",
            bordercolor=NEON_CYAN,
            font_color="#e8f7ff",
        ),
        hovermode="x unified",
        margin=dict(l=24, r=24, t=64, b=24),
    )
    fig.update_xaxes(
        gridcolor="rgba(140, 161, 184, 0.14)",
        zerolinecolor="rgba(140, 161, 184, 0.25)",
    )
    fig.update_yaxes(
        gridcolor="rgba(140, 161, 184, 0.14)",
        zerolinecolor="rgba(140, 161, 184, 0.25)",
    )
    return fig


def plot_historical_demand(df):
    fig = px.line(df, x='timestamp', y='demand', title='Historical Electricity Demand Over Time',
                  labels={'timestamp': 'Timestamp', 'demand': 'Demand (kW)'},
                  color_discrete_sequence=[NEON_CYAN])
    fig.update_traces(line=dict(width=3), fill="tozeroy", fillcolor="rgba(0, 245, 255, 0.08)")
    return _apply_dark_theme(fig)

def plot_actual_vs_predicted(df_actual, df_pred, title="Actual vs Predicted Electricity Demand"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_actual['timestamp'], y=df_actual['demand'], mode='lines', name='Actual Demand', line=dict(color=NEON_CYAN, width=3)))
    fig.add_trace(go.Scatter(x=df_pred['timestamp'], y=df_pred['forecast_demand'], mode='lines', name='Forecasted Demand', line=dict(color=NEON_GREEN, width=3, dash='dash')))
    fig.update_layout(title=title, xaxis_title='Timestamp', yaxis_title='Demand (kW)')
    return _apply_dark_theme(fig)

def plot_feature_importance(feature_names, importances):
    fig = px.bar(x=importances, y=feature_names, orientation='h', title='Feature Importance',
                 labels={'x': 'Importance Score', 'y': 'Features'},
                 color=importances,
                 color_continuous_scale=[NEON_PURPLE, NEON_PINK, NEON_CYAN])
    fig.update_layout(yaxis={'categoryorder':'total ascending'}, coloraxis_showscale=False)
    return _apply_dark_theme(fig)

def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    fig = px.scatter(x=y_pred, y=residuals, title='Residual Analysis (Predicted vs Residuals)',
                     labels={'x': 'Predicted Demand', 'y': 'Residual (Actual - Predicted)'},
                     color_discrete_sequence=[NEON_PINK])
    fig.update_traces(marker=dict(size=8, opacity=0.8, line=dict(color=NEON_CYAN, width=0.5)))
    fig.add_hline(y=0, line_dash="dash", line_color=NEON_GREEN)
    return _apply_dark_theme(fig)