import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

def cointegration_test(series1: pd.Series, series2: pd.Series):
    """
    Performs a cointegration test (e.g., Engle-Granger) on two time series.
    Returns the test statistic and p-value.
    """
    diff = series1 - series2
    result = adfuller(diff.dropna())
    test_stat, p_value = result[0], result[1]
    return test_stat, p_value

def generate_trading_signals(spread: pd.Series, threshold: float = 1.0):
    """
    Generates buy and sell signals based on the spread's deviation from its mean.
    """
    mean = spread.mean()
    std = spread.std()
    signals = pd.Series(index=spread.index, data=np.nan)
    # Buy signal: spread is significantly above the mean
    signals[spread > mean + threshold * std] = -1  # short spread
    # Sell signal: spread is significantly below the mean
    signals[spread < mean - threshold * std] = 1   # long spread
    return signals

def execute_trading_strategy(df: pd.DataFrame, spread_col: str = "spread", threshold: float = 1.0):
    """
    Executes the trading strategy based on the spread signals.
    Prints key outputs such as entry/exit signals and profit calculation.
    """
    spread = df[spread_col]
    signals = generate_trading_signals(spread, threshold)
    # For demonstration, we assume profit is the spread difference at signal reversals.
    positions = signals.fillna(method="ffill").fillna(0)
    profit = (positions.shift(-1) * (spread.diff())).fillna(0)
    total_profit = profit.sum()
    print("Total profit:", total_profit)
    return signals, total_profit

if __name__ == "__main__":
    # Dummy test for the trading strategy.
    import pandas as pd
    dates = pd.date_range("2020-01-01", periods=100)
    dummy_spread = pd.Series(data=np.random.randn(100).cumsum(), index=dates)
    df_dummy = pd.DataFrame({"spread": dummy_spread})
    signals, profit = execute_trading_strategy(df_dummy)