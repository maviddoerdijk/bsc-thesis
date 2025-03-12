import pandas as pd
import numpy as np

def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    tr = high_low.combine(high_close, max).combine(low_close, max)
    atr = tr.rolling(window=period).mean()
    return atr

def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return pd.DataFrame({"MACD": macd_line, "Signal": signal_line})

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a set of technical indicators to the DataFrame.
    """
    df = df.copy()
    df["RSI"] = compute_rsi(df["Close"])
    df["ATR"] = compute_atr(df)
    macd_df = compute_macd(df["Close"])
    df = df.join(macd_df)
    # Other indicators (MFI, ADL, VPT, Bollinger Bands, ADX, EMA, Log Return) could be added similarly
    df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))
    return df

if __name__ == "__main__":
    # Quick test if needed
    import data.data_collection as dc
    raw_data = dc.collect_data()
    df_features = add_technical_indicators(raw_data)
    print(df_features.head())