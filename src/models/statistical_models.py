import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.vector_ar.var_model import VAR
from pykalman import KalmanFilter
# from fbprophet import Prophet

def run_arima(series: pd.Series, order=(1,0,0)):
    """
    Fits an ARIMA model and forecasts one step ahead.
    """
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=1)
    return forecast, model_fit

def run_var(df: pd.DataFrame, lags: int = 1):
    """
    Fits a VAR model on multiple time series.
    """
    model = VAR(df)
    model_fit = model.fit(lags)
    forecast = model_fit.forecast(model_fit.endog, steps=1)
    return forecast, model_fit

def run_kalman(series: pd.Series):
    """
    Uses the Kalman filter to estimate and forecast the series.
    """
    kf = KalmanFilter(initial_state_mean=series.iloc[0], n_dim_obs=1)
    state_means, _ = kf.filter(series.values)
    # For a simple forecast, we use the last state as forecast
    forecast = state_means[-1]
    return forecast, kf

# def run_prophet(df: pd.DataFrame):
#     """
#     Fits Facebook Prophet to forecast the time series.
#     Assumes df has columns 'ds' and 'y'.
#     """
#     prophet_df = df.reset_index().rename(columns={"Date": "ds", df.columns[-1]: "y"})
#     m = Prophet()
#     m.fit(prophet_df)
#     future = m.make_future_dataframe(periods=1)
#     forecast = m.predict(future)
#     return forecast, m

if __name__ == "__main__":
    # Test on a sample time series (this would be replaced with spread series)
    import data.data_collection as dc
    raw_data = dc.collect_data()
    close_series = raw_data["Close"].dropna()
    fc, arima_model = run_arima(close_series)
    print("ARIMA forecast:", fc)