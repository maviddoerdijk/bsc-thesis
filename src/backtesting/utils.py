import numpy as np
import itertools
from backtesting.trading_strategy import trade

def calculate_return_uncertainty(S1, S2, spread_pred_series, position_thresholds=None, clearing_thresholds=None,
                          long_windows=None, short_windows=None, yearly_trading_days=252, return_for_plotting=False):
  if position_thresholds is not None and clearing_thresholds is not None:
      threshold_combinations = list(itertools.product(position_thresholds, clearing_thresholds))
      param_type = 'thresholds'
  elif long_windows is not None and short_windows is not None:
      threshold_combinations = list(itertools.product(long_windows, short_windows))
      param_type = 'windows'
  else:
      raise ValueError("Must specify either (position_thresholds and clearing_thresholds) or (long_windows and short_windows)")

  all_returns = []

  for a, b in threshold_combinations:
      if param_type == 'thresholds':
          returns = trade(
              S1=S1,
              S2=S2,
              spread=spread_pred_series,
              window_long=30,
              window_short=5,
              position_threshold=a,
              clearing_threshold=b
          )
          # print(f"Returns for (pt={a},ct={b}) -> {returns[-1]}")
      else:
          returns = trade(
              S1=S1,
              S2=S2,
              spread=spread_pred_series,
              window_long=a,
              window_short=b,
              position_threshold=0.8,
              clearing_threshold=0.2
          )
          # print(f"Returns for (wl={a},ws={b}) -> {returns[-1]}")

      all_returns.append(returns)

  # turn into numpy
  returns_array = np.vstack([np.array(r) for r in all_returns])

  # mean and stdev for plotting
  mean_returns = returns_array.mean(axis=0)
  std_returns = returns_array.std(axis=0)
  std_dev_pct = (std_returns / mean_returns[0]) * 100
  if return_for_plotting:
    return np.vstack([np.array(r) for r in all_returns]), param_type
  return ((mean_returns[-1] / mean_returns[0])**(yearly_trading_days / len(mean_returns)) - 1), ((std_returns[-1] / mean_returns[0]) * np.sqrt(yearly_trading_days / len(mean_returns)))
