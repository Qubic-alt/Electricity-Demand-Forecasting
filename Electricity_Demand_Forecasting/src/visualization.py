import plotly.express as px
import plotly.graph_objects as go


def plot_historical_demand(df):
    fig = px.line(df, x='timestamp', y='demand', title='Historical Electricity Demand Over Time',
                  labels={'timestamp': 'Timestamp', 'demand': 'Demand (kW)'})
    fig.update_layout(template='plotly_white', hovermode='x unified')
    return fig


def plot_actual_vs_predicted(df_actual, df_pred, title="Actual vs Predicted Electricity Demand"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_actual['timestamp'], y=df_actual['demand'], mode='lines', name='Actual Demand', line=dict(color='#1e3d59', width=2)))
    fig.add_trace(go.Scatter(x=df_pred['timestamp'], y=df_pred['forecast_demand'], mode='lines', name='Forecasted Demand', line=dict(color='#17b978', width=2, dash='dash')))
    fig.update_layout(title=title, xaxis_title='Timestamp', yaxis_title='Demand (kW)', template='plotly_white', hovermode='x unified')
    return fig


def plot_feature_importance(feature_names, importances):
    fig = px.bar(x=importances, y=feature_names, orientation='h', title='Feature Importance',
                 labels={'x': 'Importance Score', 'y': 'Features'})
    fig.update_layout(template='plotly_white', yaxis={'categoryorder': 'total ascending'})
    return fig


def plot_residuals(y_true, y_pred):
    residuals = y_true - y_pred
    fig = px.scatter(x=y_pred, y=residuals, title='Residual Analysis (Predicted vs Residuals)',
                     labels={'x': 'Predicted Demand', 'y': 'Residual (Actual - Predicted)'})
    fig.add_hline(y=0, line_dash="dash", line_color="red")
    fig.update_layout(template='plotly_white')
    return fig
