# main.py
import pandas as pd
import numpy as np

# Data collection & preprocessing
from data import data_collection as dc
from preprocessing import data_preprocessing as dp

# Models
from models import statistical_models as sm
from models import lstm_model as lm

# Trading strategy
from backtesting import trading_strategy as ts

# Visualization
from utils import visualization as viz

def main():
    # 1. Data Collection
    raw_data = dc.collect_data()
    print("Raw data shape:", raw_data.shape)
    
    # 2. Data Preprocessing
    processed_data = dp.preprocess_data(raw_data)
    print("Processed data shape:", processed_data.shape)
    
    # 3. Select a pair and calculate spread (example using two ETFs)
    # For demonstration, assume data for two tickers are available in the same DataFrame.
    etf1 = processed_data[processed_data["Ticker"]=="IVE"]
    etf2 = processed_data[processed_data["Ticker"]=="IJJ"]
    # Align dates and calculate spread
    merged = pd.merge(etf1, etf2, on="Date", suffixes=("_IVE", "_IJJ"))
    merged["spread"] = merged["Close_IVE"] - merged["Close_IJJ"]
    
    # 4. Run statistical models on the spread
    # ARIMA forecast example:
    fc_arima, _ = sm.run_arima(merged["spread"])
    print("ARIMA forecast for spread (1-step ahead):", fc_arima.values)
    
    # 5. Train LSTM model on the spread
    model, history = lm.train_lstm(merged["spread"])
    viz.plot_loss(history, title="LSTM Training Loss")
    
    # 6. Trading Strategy Execution
    signals, total_profit = ts.execute_trading_strategy(merged, spread_col="spread")
    print("Trading signals generated. Total Profit:", total_profit)
    
    # 7. Plotting examples:
    viz.plot_time_series(merged, ["Close_IVE", "Close_IJJ"], title="Raw Price Data")
    viz.plot_time_series(merged, ["spread"], title="Spread Time Series")
    
    # Further plots: heatmaps, bar charts comparing model performance etc. can be added here.
    
if __name__ == "__main__":
    main()