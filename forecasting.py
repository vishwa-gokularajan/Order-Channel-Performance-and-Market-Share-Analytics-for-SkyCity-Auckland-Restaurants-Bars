"""
SkyCity Auckland Analytics - Time-Series Forecasting Engine
Implements Simple Moving Average, Exponential Smoothing, and ARIMA models for order volume and revenue forecasting.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

def prepare_daily_timeseries(df: pd.DataFrame, metric: str = 'Net_Revenue') -> pd.DataFrame:
    """Aggregates transactional orders into continuous daily time-series."""
    if df.empty or metric not in df.columns:
        return pd.DataFrame(columns=[metric])
    df_daily = df.groupby('Order_Date')[metric].sum().reset_index()
    df_daily['Order_Date'] = pd.to_datetime(df_daily['Order_Date'])
    df_daily = df_daily.sort_values('Order_Date').set_index('Order_Date')
    # Reindex to fill any missing calendar dates with 0
    full_idx = pd.date_range(start=df_daily.index.min(), end=df_daily.index.max(), freq='D')
    df_daily = df_daily.reindex(full_idx, fill_value=0.0)
    df_daily.index.name = 'Order_Date'
    return df_daily

def calculate_forecast_metrics(y_true, y_pred) -> dict:
    """Computes MAE, RMSE, and MAPE metrics."""
    if len(y_true) == 0 or len(y_pred) == 0:
        return {'MAE': 0.0, 'RMSE': 0.0, 'MAPE': 0.0}
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-5))) * 100
    return {
        'MAE': round(float(mae), 2),
        'RMSE': round(float(rmse), 2),
        'MAPE': round(float(mape), 2)
    }

def forecast_moving_average(ts_data: pd.Series, horizon_days: int = 30, window: int = 14) -> tuple[pd.Series, pd.DataFrame]:
    """Simple Moving Average Baseline Forecast."""
    train = ts_data.iloc[:-horizon_days]
    test = ts_data.iloc[-horizon_days:]
    
    last_ma = train.iloc[-window:].mean()
    future_dates = pd.date_range(start=test.index[0], periods=horizon_days, freq='D')
    forecast_series = pd.Series([last_ma] * horizon_days, index=future_dates)
    
    metrics = calculate_forecast_metrics(test.values, forecast_series.values)
    return forecast_series, metrics

def forecast_exponential_smoothing(ts_data: pd.Series, horizon_days: int = 30) -> tuple[pd.Series, pd.DataFrame]:
    """Holt-Winters Exponential Smoothing Forecast."""
    train = ts_data.iloc[:-horizon_days]
    test = ts_data.iloc[-horizon_days:]
    
    model = ExponentialSmoothing(train, seasonal='add', seasonal_periods=7).fit()
    forecast_vals = model.forecast(horizon_days)
    
    metrics = calculate_forecast_metrics(test.values, forecast_vals.values)
    return forecast_vals, metrics

def forecast_arima(ts_data: pd.Series, horizon_days: int = 30, order: tuple = (1, 1, 1)) -> tuple[pd.Series, dict, pd.DataFrame]:
    """ARIMA Time-Series Model Forecast with Confidence Intervals."""
    train = ts_data.iloc[:-horizon_days]
    test = ts_data.iloc[-horizon_days:]
    
    model = ARIMA(train, order=order).fit()
    forecast_res = model.get_forecast(steps=horizon_days)
    forecast_series = forecast_res.predicted_mean
    conf_int = forecast_res.conf_int()
    
    metrics = calculate_forecast_metrics(test.values, forecast_series.values)
    return forecast_series, conf_int, metrics

def plot_time_series_forecast(ts_data: pd.Series, forecast_series: pd.Series, conf_int: pd.DataFrame = None, title: str = "Time-Series Revenue Forecast"):
    """Creates a high-quality Plotly forecast visualization with confidence interval shading."""
    fig = go.Figure()
    
    # Historical Actuals
    fig.add_trace(go.Scatter(
        x=ts_data.index, y=ts_data.values,
        mode='lines', name='Historical Actuals',
        line=dict(color='#3B82F6', width=2)
    ))
    
    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_series.index, y=forecast_series.values,
        mode='lines+markers', name='Predicted Forecast',
        line=dict(color='#10B981', width=3, dash='dash')
    ))
    
    # Confidence Interval Shading
    if conf_int is not None:
        fig.add_trace(go.Scatter(
            x=forecast_series.index.tolist() + forecast_series.index.tolist()[::-1],
            y=conf_int.iloc[:, 1].tolist() + conf_int.iloc[:, 0].tolist()[::-1],
            fill='todense',
            fillcolor='rgba(16, 185, 129, 0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name='95% Confidence Bounds'
        ))
        
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Value",
        template='plotly_dark',
        hovermode="x unified"
    )
    return fig
