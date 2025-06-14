import numpy as np
import os
from sklearn.metrics import mean_squared_error
from pykalman import KalmanFilter
import pandas as pd
from typing import Dict, Any, Sequence
import torch
import random

# custom imports
from backtesting.trading_strategy import get_gt_yoy_returns_test_dev
from backtesting.utils import calculate_return_uncertainty

def acc_metric(true_value: Sequence[np.ndarray],
               predicted_value: Sequence[np.ndarray]) -> float:
    """
    Legacy RMSE scorer used in the original script:
    both inputs are **lists of 1-D arrays** (len == look_back).
    """
    acc_met = 0.0
    m = len(true_value)
    for i in range(m):
        acc_met += mean_squared_error(true_value[i], predicted_value[i])
    return np.sqrt(acc_met / m)

def kalman_filter_average(x: pd.Series,
                          transition_cov: float = 0.01,
                          obs_cov: float = 1.) -> pd.Series:
    kf = KalmanFilter(transition_matrices=[1],
                      observation_matrices=[1],
                      initial_state_mean=0.,
                      initial_state_covariance=1.,
                      transition_covariance=transition_cov,
                      observation_covariance=obs_cov)
    state_means, _ = kf.filter(x.values)
    return pd.Series(state_means.flatten(), index=x.index)

def kalman_filter_regression(x: pd.Series,
                             y: pd.Series,
                             delta: float = 1e-3,
                             obs_cov: float = 2.) -> np.ndarray:
    trans_cov = delta / (1.0 - delta) * np.eye(2)
    obs_mat = np.expand_dims(np.vstack([x, np.ones(len(x))]).T, axis=1)

    kf = KalmanFilter(n_dim_obs=1, n_dim_state=2,
                      transition_matrices=np.eye(2),
                      observation_matrices=obs_mat,
                      initial_state_mean=np.zeros(2),
                      initial_state_covariance=np.eye(2),
                      transition_covariance=trans_cov,
                      observation_covariance=obs_cov)

    state_means, _ = kf.filter(y.values)
    return state_means # has same length, so no shortening needed here
  
def execute_kalman_workflow(
  pairs_timeseries: pd.DataFrame,
  target_col: str = "Spread_Close",
  col_s1: str = "S1_close",
  col_s2: str = "S2_close",
  train_frac: float = 0.90,
  dev_frac: float = 0.05,
  seed: int = 3178749, # for reproducibility, my student number
  look_back: int = 1,
  yearly_trading_days: int = 252,
  ## optimized hyperparams ##
  delta: float = 1e-3,
  obs_cov_reg: float = 2.,
  trans_cov_avg: float = 0.01,
  obs_cov_avg: float = 1.,
  ## optimized hyperparams ##
  return_datasets: bool = False,
  verbose: bool = True,
  result_parent_dir: str = "data/results",
  filename_base: str = "data_begindate_enddate_hash.pkl", # use `_get_filename(startDateStr, endDateStr, instrumentIds)`
  pair_tup_str: str = "(?,?)", # Used for showing which tuple was used in plots, example: "(QQQ, SPY)"
):
  # Set seeds
  torch.manual_seed(seed)
  np.random.seed(seed)
  random.seed(seed)

  # For GPU (if used)
  if torch.cuda.is_available():
      torch.cuda.manual_seed(seed)
      torch.cuda.manual_seed_all(seed)
      torch.backends.cudnn.deterministic = True
      torch.backends.cudnn.benchmark = False  # Might slow down, but ensures determinism
      
  # Check whether everything is present as expected (good practice, and gives useful exceptions)
  required = {col_s1, col_s2}
  if not required.issubset(pairs_timeseries.columns):
      raise KeyError(f"pairs_timeseries must contain {required}")

  total_len = len(pairs_timeseries)
  train_size = int(total_len * train_frac)
  dev_size   = int(total_len * dev_frac)
  test_size  = total_len - train_size - dev_size # not used, but for clarity

  train_univariate = pairs_timeseries.iloc[:train_size][target_col]
  dev_univariate = pairs_timeseries.iloc[train_size:train_size + dev_size][target_col]
  test_univariate = pairs_timeseries.iloc[train_size + dev_size:][target_col]
  
  train_multivariate = pairs_timeseries.iloc[:train_size]
  dev_multivariate = pairs_timeseries.iloc[train_size:train_size + dev_size]
  test_multivariate = pairs_timeseries.iloc[train_size + dev_size:]  

  if verbose:
      print(f"Split sizes — train: {len(train_univariate)}, dev: {len(dev_univariate)}, test: {len(test_univariate)}")
      
  def create_sequences(series):
      # series: pd.Series
      mean = series.mean()
      std = series.std()
      series_scaled = (series - mean) / (std + 1e-8)
      return series_scaled, mean, std

  train_scaled, train_mean, train_std = create_sequences(train_multivariate)  
  dev_scaled, _, _ = create_sequences(dev_multivariate)
  test_scaled, _, _ = create_sequences(test_multivariate) # Note: to prevent data leakage, means and std's of test and dev may not be used.

  pairs_timeseries_scaled = pd.concat([train_scaled, dev_scaled, test_scaled])

  # get beta_t, the Kalman-filtered regression coefficients
  # Note: since the three time series are scaled independently, there is not any lookahead bias
  beta_t = kalman_filter_regression(
      kalman_filter_average(pairs_timeseries_scaled[col_s1],
                            transition_cov=trans_cov_avg,
                            obs_cov=obs_cov_avg),
      kalman_filter_average(pairs_timeseries_scaled[col_s2],
                            transition_cov=trans_cov_avg,
                            obs_cov=obs_cov_avg),
      delta=delta,
      obs_cov=obs_cov_reg
  )[:, 0]

  normed_predictions = pairs_timeseries_scaled[col_s1] + pairs_timeseries_scaled[col_s2] * beta_t

  if len(beta_t) != len(pairs_timeseries_scaled):
    raise Exception("Sanity check failed: len(beta_t) != len(pairs_timeseries_scaled)")

  forecast_train_normed = normed_predictions[:len(train_scaled)].to_numpy() # Though the variable `forecast_train` is never directly used as a variable, the data for it WAS used, during kalman filter averaging and regression
  forecast_dev_normed  = normed_predictions[len(train_scaled):len(train_scaled) + len(dev_scaled)].to_numpy()
  forecast_test_normed = normed_predictions[-len(test_scaled):].to_numpy()

  if look_back == 1:
      # Calculate mse values
      groundtruth_test = pairs_timeseries[target_col].iloc[-len(test_multivariate):]
      # format into wanted form for `acc_metric` function
      groundtruth_test_formatted = np.array([[v] for v in groundtruth_test])
      forecast_test_original_scale = forecast_test_normed * train_std + train_mean
      forecast_test_formatted = np.array([[v] for v in forecast_test_original_scale])

      test_mse = acc_metric(groundtruth_test_formatted, forecast_test_formatted)
      test_var = np.var(groundtruth_test)
      test_nmse = test_mse / test_var if test_var != 0 else 0.0

      # also for validation
      groundtruth_dev = pairs_timeseries[target_col].iloc[len(train_multivariate):len(train_multivariate) + len(dev_multivariate)]
      groundtruth_dev_formatted = np.array([[v] for v in groundtruth_dev])
      forecast_dev_original_scale = forecast_dev_normed * train_std + train_mean
      forecast_dev_formatted = np.array([[v] for v in forecast_dev_original_scale])

      val_mse = acc_metric(groundtruth_dev_formatted, forecast_dev_formatted)
      val_var = np.var(groundtruth_dev)
      val_nmse = val_mse / val_var if val_var != 0 else 0.
  else:
      raise NotImplementedError("Warning: look_back > 1 not yet implemented. Returning None for mse.")

  ### TRADING ###
  # calculate std_dev_pct using the logic from plot_with_uncertainty. Then put that into two separate functions: calculate_yoy_uncertainty and a version of plot_with_uncertainty that uses calculate_yoy_uncertainty
  # position threshold (2.00-4.00), clearing threshold (0.30-0.70)
  min_position = 2.00
  max_position = 4.00
  min_clearing = 0.30
  max_clearing = 0.70
  position_thresholds = np.linspace(min_position, max_position, num=10)
  clearing_thresholds = np.linspace(min_clearing, max_clearing, num=10)

  test_index = test_multivariate.index
  forecast_test_series = pd.Series(forecast_test_original_scale, index=test_index)
  test_s1 = test_multivariate['S1_close']
  test_s2 = test_multivariate['S2_close']

  yoy_mean, yoy_std = calculate_return_uncertainty(test_s1, test_s2, forecast_test_series, position_thresholds=position_thresholds, clearing_thresholds=clearing_thresholds, yearly_trading_days=yearly_trading_days)
  # calculate the strategy returns if we were to feed the groundtruth values to the `trade` func. If the ground truth returns are lower, it seems likely there is something wrong with the `trade` func (but not certain! Probability applies here).
  # forecast_test_shortened = forecast_test[:len(testY_untr)]
  # spread_pred_series = pd.Series(forecast_test_shortened, index=index_shortened)
  gt_test_series = pd.Series(test_multivariate['Spread_Close'].values, index=test_index)
  output = get_gt_yoy_returns_test_dev(pairs_timeseries, dev_frac, train_frac, look_back=0, yearly_trading_day=yearly_trading_days)
  gt_yoy, gt_yoy_for_dev_dataset = output['gt_yoy_test'], output['gt_yoy_dev']

  current_result_dir = filename_base.replace(".pkl", "_kalman")  
  result_dir = os.path.join(result_parent_dir, current_result_dir)
  if not os.path.exists(result_dir):
      os.makedirs(result_dir)

  output: Dict[str, Any] = dict(
      val_mse=val_nmse,
      test_mse=test_nmse,
      yoy_mean=yoy_mean,
      yoy_std=yoy_std,
      gt_yoy=gt_yoy,
      result_parent_dir=result_parent_dir,
  )
  
  results_str = f"""
  Validation MSE: {output['val_mse']}
  Test MSE: {output['test_mse']}
  YOY Returns: {output['yoy_mean'] * 100:.2f}%
  YOY Std: +- {output['yoy_std'] * 100:.2f}%
  GT Yoy: {output['gt_yoy'] * 100:.2f}%
  Plot filepath parent dir: {output['result_parent_dir']}
  pair_tup_str: {pair_tup_str}
  """
  with open(os.path.join(result_dir, "results.txt"), "w") as f:
      f.write(results_str)
  if verbose:
    print(results_str)
  if return_datasets:
      output.update(
          dict(
            test_s1_shortened=test_s1, 
            test_s2_shortened=test_s2, 
            forecast_test_shortened_series=forecast_test_series, 
            gt_test_shortened_series=gt_test_series
          )
      )
  return output
