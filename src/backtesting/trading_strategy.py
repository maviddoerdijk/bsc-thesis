import pandas as pd

def trade(
        S1: pd.Series,
        S2: pd.Series,
        spread: pd.Series, # model-predicted spread for the strategy
        window_long: int,
        window_short: int,
        position_threshold: float = 1.0,
        clearing_threshold: float = 0.5,
        risk_fraction: float = 0.1 # could be used again
    ):   
    if len(spread) != len(S1) or len(spread) != len(S2):
        raise ValueError("Length of S1, S2, and spread must be the same")
    # Compute rolling mean and rolling standard deviation

    ma_long = spread.rolling(window=window_long, center=False).mean()
    ma_short = spread.rolling(window=window_short, center=False).mean()
    std = spread.rolling(window=window_short, center=False).std()
    zscore = (ma_long - ma_short)/std

    # Calculate initial cash based on average range of S1, S2 and Spread_Close, as these also determine the size of the trades
    s2_spread = max(S2) - min(S2)
    s1_spread = max(S1) - min(S1)
    spread_spread = max(spread) - min(spread)
    avg_spread = (s2_spread + s1_spread + spread_spread) / 3
    initial_cash = avg_spread * len(spread) # the absolute returns are correlated to the length of the spread, times the average range.

    # Simulate trading
    # Start with no money and no positions
    cash = initial_cash # initial cash amount, perhaps not hardcoded in the future
    qty_s1 = 0
    qty_s2 = 0
    returns = [initial_cash]

    for i in range(len(spread)):
        # Sell short if the z-score is > 1
        if zscore.iloc[i] > position_threshold:
            # print(f"[NEW] Step {i}: SELL SHORT, z={zscore.iloc[i]:.2f}, S1={S1.iloc[i]:.2f}, S2={S2.iloc[i]:.2f}, spread={spread.iloc[i]:.2f}, cash={cash:.2f}, qty_s1={qty_s1}, qty_s2={qty_s2}")
            cash += S1.iloc[i] - S2.iloc[i] * spread.iloc[i]
            qty_s1 -= 1
            qty_s2 += spread.iloc[i]
        # Buy long if the z-score is < 1
        elif zscore.iloc[i] < -position_threshold:
            # print(f"[NEW] Step {i}: BUY LONG, z={zscore.iloc[i]:.2f}, S1={S1.iloc[i]:.2f}, S2={S2.iloc[i]:.2f}, spread={spread.iloc[i]:.2f}, cash={cash:.2f}, qty_s1={qty_s1}, qty_s2={qty_s2}")
            cash -= S1.iloc[i] - S2.iloc[i] * spread.iloc[i]
            qty_s1 += 1
            qty_s2 -= spread.iloc[i]
        # Clear positions if the z-score between -.5 and .5
        elif abs(zscore.iloc[i]) < clearing_threshold:
            # print(f"[NEW] Step {i}: CLEAR POSITION, z={zscore.iloc[i]:.2f}, S1={S1.iloc[i]:.2f}, S2={S2.iloc[i]:.2f}, spread={spread.iloc[i]:.2f}, cash={cash:.2f}, qty_s1={qty_s1}, qty_s2={qty_s2}")
            cash += qty_s1 * S1.iloc[i] - S2.iloc[i] * qty_s2
            qty_s1 = 0
            qty_s2 = 0
        returns.append(cash) # append the current cash value to returns
    # If at any point returns is 0, all values after that is zero
    zero_from_this_idx = -1
    for i in range(len(returns)):
        if returns[i] <= 0:
            zero_from_this_idx = i
            break
    if zero_from_this_idx > -1:
        returns[zero_from_this_idx:] = [0] * (len(returns) - zero_from_this_idx)

    # Shrink returns by a factor such that returns are not inflated.
    returns_series = pd.Series(returns)
    alpha = 0.1  # Shrinking/stretching factor
    returns_uninflated = returns_series[0] + alpha * (returns_series - returns_series[0])
    # turn back into list
    returns_uninflated = returns_uninflated.tolist()
    return returns_uninflated

def get_gt_yoy_returns_test_dev(pairs_timeseries_df, dev_frac, train_frac, look_back):
  burn_in = 30

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