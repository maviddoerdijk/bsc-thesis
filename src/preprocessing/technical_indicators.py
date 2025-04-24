import ta
import statsmodels.api as sm
import pandas as pd

def add_technical_indicators(pairs_time_series):
    pairs_timeseries_including_ta = pairs_time_series.copy()

    # add technical indicators
    # 1. Momentum Indicators
    # Relative Strength Index
    pairs_timeseries_including_ta['S1_rsi'] = ta.momentum.rsi(pairs_timeseries_including_ta['S1_close'], window=14)
    pairs_timeseries_including_ta['S2_rsi'] = ta.momentum.rsi(pairs_timeseries_including_ta['S2_close'], window=14)


    # 2. Volume Indicators
    # Money Flow Index
    pairs_timeseries_including_ta['S1_mfi'] = ta.volume.money_flow_index(pairs_timeseries_including_ta['S1_high'], pairs_timeseries_including_ta['S1_low'],
                                                        pairs_timeseries_including_ta['S1_close'], pairs_timeseries_including_ta['S1_volume'], window=14)
    pairs_timeseries_including_ta['S2_mfi'] = ta.volume.money_flow_index(pairs_timeseries_including_ta['S2_high'], pairs_timeseries_including_ta['S2_low'],
                                                        pairs_timeseries_including_ta['S2_close'], pairs_timeseries_including_ta['S2_volume'], window=14)
    # Accumulation/Distribution Index (ADI)
    pairs_timeseries_including_ta['S1_adi'] = ta.volume.acc_dist_index(pairs_timeseries_including_ta['S1_high'], pairs_timeseries_including_ta['S1_low'], pairs_timeseries_including_ta['S1_close'], pairs_timeseries_including_ta['S1_volume'])
    pairs_timeseries_including_ta['S2_adi'] = ta.volume.acc_dist_index(pairs_timeseries_including_ta['S2_high'], pairs_timeseries_including_ta['S2_low'], pairs_timeseries_including_ta['S2_close'], pairs_timeseries_including_ta['S2_volume'])
    # Volume-price trend (VPT)
    pairs_timeseries_including_ta['S1_vpt'] = ta.volume.volume_price_trend(pairs_timeseries_including_ta['S1_close'], pairs_timeseries_including_ta['S1_volume'])
    pairs_timeseries_including_ta['S2_vpt'] = ta.volume.volume_price_trend(pairs_timeseries_including_ta['S2_close'], pairs_timeseries_including_ta['S2_volume'])

    # 3. Volatility Indicators
    # Average True Range (ATR)
    pairs_timeseries_including_ta['S1_atr'] = ta.volatility.average_true_range(pairs_timeseries_including_ta['S1_high'], pairs_timeseries_including_ta['S1_low'],
                                                            pairs_timeseries_including_ta['S1_close'], window=14)
    pairs_timeseries_including_ta['S2_atr'] = ta.volatility.average_true_range(pairs_timeseries_including_ta['S2_high'], pairs_timeseries_including_ta['S2_low'],
                                                            pairs_timeseries_including_ta['S2_close'], window=14)
    # Bollinger Bands (BB) N-period simple moving average (MA)
    pairs_timeseries_including_ta['S1_bb_ma'] = ta.volatility.bollinger_mavg(pairs_timeseries_including_ta['S1_close'], window=20)
    pairs_timeseries_including_ta['S2_bb_ma'] = ta.volatility.bollinger_mavg(pairs_timeseries_including_ta['S2_close'], window=20)

    # 4. Trend Indicators
    # Average Directional Movement Index (ADX)
    pairs_timeseries_including_ta['S1_adx'] = ta.trend.adx(pairs_timeseries_including_ta['S1_high'], pairs_timeseries_including_ta['S1_low'], pairs_timeseries_including_ta['S1_close'], window=14)
    pairs_timeseries_including_ta['S2_adx'] = ta.trend.adx(pairs_timeseries_including_ta['S2_high'], pairs_timeseries_including_ta['S2_low'], pairs_timeseries_including_ta['S2_close'], window=14)
    # Exponential Moving Average
    pairs_timeseries_including_ta['S1_ema'] = ta.trend.ema_indicator(pairs_timeseries_including_ta['S1_close'], window=14)
    pairs_timeseries_including_ta['S2_ema'] = ta.trend.ema_indicator(pairs_timeseries_including_ta['S2_close'], window=14)
    # Moving Average Convergence Divergence (MACD)
    pairs_timeseries_including_ta['S1_macd'] = ta.trend.macd(pairs_timeseries_including_ta['S1_close'], window_fast=14, window_slow=30)
    pairs_timeseries_including_ta['S2_macd'] = ta.trend.macd(pairs_timeseries_including_ta['S2_close'], window_fast=14, window_slow=30)

    # 5. Other Indicators
    # Daily Log Return (DLR)
    pairs_timeseries_including_ta['S1_dlr'] = ta.others.daily_log_return(pairs_timeseries_including_ta['S1_close'])
    pairs_timeseries_including_ta['S2_dlr'] = ta.others.daily_log_return(pairs_timeseries_including_ta['S2_close'])
    return pairs_timeseries_including_ta

def combine_pairs_data(data_close, data_open, data_high, data_low, data_vol, ticker1, ticker2):
    df = pd.DataFrame({
        'S1_close': data_close[ticker1], 'S2_close': data_close[ticker2],
        'S1_open': data_open[ticker1], 'S2_open': data_open[ticker2],
        'S1_high': data_high[ticker1], 'S2_high': data_high[ticker2],
        'S1_low': data_low[ticker1], 'S2_low': data_low[ticker2],
        'S1_volume': data_vol[ticker1], 'S2_volume': data_vol[ticker2],
    })

    # Technical Indicators
    df['S1_rsi'] = ta.momentum.rsi(df['S1_close'], window=14)
    df['S2_rsi'] = ta.momentum.rsi(df['S2_close'], window=14)

    df['S1_mfi'] = ta.volume.money_flow_index(df['S1_high'], df['S1_low'], df['S1_close'], df['S1_volume'], window=14)
    df['S2_mfi'] = ta.volume.money_flow_index(df['S2_high'], df['S2_low'], df['S2_close'], df['S2_volume'], window=14)

    df['S1_adi'] = ta.volume.acc_dist_index(df['S1_high'], df['S1_low'], df['S1_close'], df['S1_volume'])
    df['S2_adi'] = ta.volume.acc_dist_index(df['S2_high'], df['S2_low'], df['S2_close'], df['S2_volume'])

    df['S1_vpt'] = ta.volume.volume_price_trend(df['S1_close'], df['S1_volume'])
    df['S2_vpt'] = ta.volume.volume_price_trend(df['S2_close'], df['S2_volume'])

    df['S1_atr'] = ta.volatility.average_true_range(df['S1_high'], df['S1_low'], df['S1_close'], window=14)
    df['S2_atr'] = ta.volatility.average_true_range(df['S2_high'], df['S2_low'], df['S2_close'], window=14)

    df['S1_bb_ma'] = ta.volatility.bollinger_mavg(df['S1_close'], window=20)
    df['S2_bb_ma'] = ta.volatility.bollinger_mavg(df['S2_close'], window=20)

    df['S1_adx'] = ta.trend.adx(df['S1_high'], df['S1_low'], df['S1_close'], window=14)
    df['S2_adx'] = ta.trend.adx(df['S2_high'], df['S2_low'], df['S2_close'], window=14)

    df['S1_ema'] = ta.trend.ema_indicator(df['S1_close'], window=14)
    df['S2_ema'] = ta.trend.ema_indicator(df['S2_close'], window=14)

    df['S1_macd'] = ta.trend.macd(df['S1_close'], window_fast=14, window_slow=30)
    df['S2_macd'] = ta.trend.macd(df['S2_close'], window_fast=14, window_slow=30)

    df['S1_dlr'] = ta.others.daily_log_return(df['S1_close'])
    df['S2_dlr'] = ta.others.daily_log_return(df['S2_close'])

    # Spreads via regression
    alpha_c = -sm.OLS(df['S1_close'], df['S2_close']).fit().params[0]
    alpha_o = -sm.OLS(df['S1_open'], df['S2_open']).fit().params[0]
    alpha_h = -sm.OLS(df['S1_high'], df['S2_high']).fit().params[0]
    alpha_l = -sm.OLS(df['S1_low'], df['S2_low']).fit().params[0]

    df['Spread_Close'] = df['S1_close'] + df['S2_close'] * alpha_c
    df['Spread_Open'] = df['S1_open'] + df['S2_open'] * alpha_o
    df['Spread_High'] = df['S1_high'] + df['S2_high'] * alpha_h
    df['Spread_Low'] = df['S1_low'] + df['S2_low'] * alpha_l

    return df