def step_1_filter_remove_nans(data_close, data_open, data_high, data_low, data_vol, data):
  d = data_close.isnull().any()
  valid_tickers = d[d == False].index

  # keep all the tickers where none of the entries are null
  data_close_filtered_1 = data_close[valid_tickers]
  data_open_filtered_1 = data_open[valid_tickers]
  data_high_filtered_1 = data_high[valid_tickers]
  data_low_filtered_1 = data_low[valid_tickers]
  data_vol_filtered_1 = data_vol[valid_tickers]
  data_original_format_filtered_1 = data['yfinance_formatted'][valid_tickers]
  return data_close_filtered_1, data_open_filtered_1, data_high_filtered_1, data_low_filtered_1, data_vol_filtered_1, data_original_format_filtered_1

def step_2_filter_liquidity(data_close_filtered_1, data_open_filtered_1, data_high_filtered_1, data_low_filtered_1, data_vol_filtered_1, data_original_format_filtered_1, liquidity_threshold = 10**5):
    avg_vols = data_vol_filtered_1.mean(axis=0)

    # find liquid tickers
    liquid_tickers = avg_vols[avg_vols > liquidity_threshold].index

    # similar to step 1, filter again using the tickers
    data_close_filtered_2 = data_close_filtered_1[liquid_tickers]
    data_open_filtered_2 = data_open_filtered_1[liquid_tickers]
    data_high_filtered_2 = data_high_filtered_1[liquid_tickers]
    data_low_filtered_2 = data_low_filtered_1[liquid_tickers]
    data_vol_filtered_2 = data_vol_filtered_1[liquid_tickers]
    data_original_format_filtered_2 = data_original_format_filtered_1[liquid_tickers]

    return data_close_filtered_2, data_open_filtered_2, data_high_filtered_2, data_low_filtered_2, data_vol_filtered_2, data_original_format_filtered_2
