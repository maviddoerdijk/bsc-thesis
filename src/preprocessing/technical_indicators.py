import ta

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