import pandas as pd

def trade(
    S1: pd.Series,
    S2: pd.Series,
    spread: pd.Series, # model-predicted spread for the strategy
    window_long: int,
    window_short: int,
    initial_cash: float = 100000,
    position_threshold: float = 1.0,
    clearing_threshold: float = 0.5,
    risk_fraction: float = 0.1
) -> list:
    ma_long = spread.rolling(window=window_long, center=False).mean()
    ma_short = spread.rolling(window=window_short, center=False).mean()
    std = spread.rolling(window=window_short, center=False).std()
    zscore = (ma_long - ma_short)/std

    # starting with initial_cash allows us to calculate the returns as a percentage of the initial cash, over a time period (YoY are not equal to QoQ for example)
    cash = initial_cash
    qty_s1 = 0
    qty_s2 = 0
    returns = [initial_cash] # note: starting with initial cash allows us to calculate the returns as a percentage of the initial cash, over a time period (YoY are not equal to QoQ for example), but it causes a problem with plotting where we extend the returns length by one. Be wary of this.
    position = 0 # 0: neutral, 1: long, -1: short
    
    if len(spread) != len(S1) or len(spread) != len(S2):
        raise ValueError("Length of S1, S2, and spread must be the same")

    for i in range(len(spread)): # each iteration of the for loop is a new time step, in this case often a single day
        price_s1 = S1.iloc[i]
        price_s2 = S2.iloc[i]
        beta = spread.iloc[i]
        equity = cash + qty_s1 * price_s1 - qty_s2 * price_s2

        # Enter short spread (short S1, long beta S2)
        if position == 0 and zscore.iloc[i] > position_threshold:
            position = -1
            position_size = equity * risk_fraction
            qty_s1 = -position_size / price_s1
            qty_s2 = (position_size * beta) / price_s2
            cash -= (qty_s1 * price_s1 - qty_s2 * price_s2)

        # Enter long spread (long S1, short beta S2)
        elif position == 0 and zscore.iloc[i] < -position_threshold:
            position = 1
            position_size = equity * risk_fraction
            qty_s1 = position_size / price_s1
            qty_s2 = - (position_size * beta) / price_s2
            cash -= (qty_s1 * price_s1 - qty_s2 * price_s2)

        # Exit to neutral when spread reverts
        elif position != 0 and abs(zscore.iloc[i]) < clearing_threshold:
            cash += qty_s1 * price_s1 - qty_s2 * price_s2
            qty_s1 = 0
            qty_s2 = 0
            position = 0

        equity = cash + qty_s1 * price_s1 - qty_s2 * price_s2
        returns.append(equity)

    return returns

def get_gt_yoy_returns_test_dev(pairs_timeseries_df, dev_frac, train_frac, look_back):
  burn_in = 20

  pairs_timeseries_df_burned_in = pairs_timeseries_df.iloc[burn_in:].copy()

  total_len = len(pairs_timeseries_df_burned_in)
  train_size = int(total_len * train_frac)
  dev_size   = int(total_len * dev_frac)
  test_size  = total_len - train_size - dev_size # not used, but for clarity

  train = pairs_timeseries_df_burned_in.iloc[:train_size]
  dev   = pairs_timeseries_df_burned_in.iloc[train_size:train_size + dev_size]
  test  = pairs_timeseries_df_burned_in.iloc[train_size + dev_size:]


  index_shortened = test.index[:len(test['Spread_Close'].values[look_back:])] # problem: test['S1_close'].iloc[look_back:] and testY_untr are the same.. So we should rather be using test
  spread_gt_series = pd.Series(test['Spread_Close'].values[look_back:], index=index_shortened)
  gt_returns_test = trade(
      S1 = test['S1_close'].iloc[look_back:],
      S2 = test['S2_close'].iloc[look_back:],
      spread = spread_gt_series,
      window_long = 30,
      window_short = 5,
      position_threshold = 3,
      clearing_threshold = 0.4
  )
  gt_yoy_test = ((gt_returns_test[-1] / gt_returns_test[0])**(365 / len(gt_returns_test)) - 1)

  index_shortened = dev.index[:len(dev['Spread_Close'].values[look_back:])]
  spread_gt_series = pd.Series(dev['Spread_Close'].values[look_back:], index=index_shortened)
  gt_returns_dev = trade(
      S1 = dev['S1_close'].iloc[look_back:],
      S2 = dev['S2_close'].iloc[look_back:],
      spread = spread_gt_series,
      window_long = 30,
      window_short = 5,
      position_threshold = 3,
      clearing_threshold = 0.4
  )
  gt_yoy_dev = ((gt_returns_dev[-1] / gt_returns_dev[0])**(365 / len(gt_returns_dev)) - 1)
  return {
      "gt_yoy_test": gt_yoy_test, 
      "gt_yoy_dev": gt_yoy_dev
  }