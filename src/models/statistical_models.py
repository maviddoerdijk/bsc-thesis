import pywt
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from pykalman import KalmanFilter
import pandas as pd
from matplotlib import pyplot as plt
from typing import Callable, Dict, Any, Optional, Tuple, Sequence, Union
import torch
import random

# custom imports
from backtesting.trading_strategy import trade
from backtesting.utils import calculate_return_uncertainty
from utils.visualization import plot_return_uncertainty, plot_comparison
from preprocessing.wavelet_denoising import wav_den


def create_dataset(mat: np.ndarray, scaler: MinMaxScaler = MinMaxScaler(feature_range=(0, 1)), look_back: int = 1
                    ) -> Tuple[np.ndarray, np.ndarray,
                                np.ndarray, np.ndarray]:
    """
    Return  (raw_X, scaled_X, raw_Y, scaled_Y),
    where Y is a 1-step shift of X[:,0].
    """ 
    dataX, dataY = [], []
    for i in range(len(mat) - look_back):
        dataX.append(mat[i, :])
        dataY.append(mat[(i + 1):(i + 1 + look_back), 0])
    return (dataX,
            scaler.fit_transform(dataX),
            dataY,
            scaler.fit_transform(dataY))

def default_normalize(series: pd.Series) -> pd.Series:
    # z-score normalization
    return (series - series.mean()) / series.std(ddof=0)

def rmse_metric(y_true: Sequence[np.ndarray],
                y_pred: Sequence[np.ndarray]) -> float:
    # Currently using RMSE
    mse = np.mean([mean_squared_error(yt, yp) for yt, yp in zip(y_true, y_pred)])
    return np.sqrt(mse)

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
    return state_means          # shape (T, 2)

def kalman_filter_regression_multivariate(X, y, delta=1e-4, obs_cov=1e-2):
    T, d = X.shape
    transition_matrix = np.eye(d)
    transition_covariance = delta * np.eye(d)
    # initially coefficients at zero
    initial_state_mean = np.zeros(d)
    initial_state_covariance = 1e3 * np.eye(d)
    observation_covariance = obs_cov
    observation_matrices = X[:, np.newaxis, :]

    kf = KalmanFilter(
        transition_matrices=transition_matrix,
        observation_matrices=observation_matrices,
        initial_state_mean=initial_state_mean,
        initial_state_covariance=initial_state_covariance,
        transition_covariance=transition_covariance,
        observation_covariance=obs_cov,
        n_dim_obs=1,
        n_dim_state=d
    )

    state_means, _ = kf.filter(y)
    return state_means

def execute_kalman_workflow(
  pair_data: pd.DataFrame,
  col_s1: str = "S1_close",
  col_s2: str = "S2_close",
  burn_in: int = 30,
  train_frac: float = 0.90,
  dev_frac: float = 0.05,
  seed: int = 3178749, # for reproducibility, my student number
  look_back: int = 1,
  denoise_fn: Optional[Callable[[pd.Series], np.ndarray]] = wav_den,
  scaler_factory: Callable[..., MinMaxScaler] = MinMaxScaler,
  scaler_kwargs: Optional[Dict[str, Any]] = {"feature_range": (0, 1)},
  normalise_fn: Callable[[pd.Series], pd.Series] = default_normalize,
  delta: float = 1e-3,
  obs_cov_reg: float = 2.,
  trans_cov_avg: float = 0.01,
  obs_cov_avg: float = 1.,
  return_datasets: bool = False,
  verbose: bool = True,
  add_technical_indicators: bool = True,
  result_parent_dir: str = "data/results",
  filename_base: str = "data_begindate_enddate_hash.pkl", # use `_get_filename(startDateStr, endDateStr, instrumentIds)`
  pair_tup_str: str = "(?,?)" # Used for showing which tuple was used in plots, example: "(QQQ, SPY)"
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
  if not required.issubset(pair_data.columns):
      raise KeyError(f"pair_data must contain {required}")

  keep_cols = [c for c in pair_data.columns if c not in ("date",)]
  df = pair_data[keep_cols].iloc[burn_in:].copy()

  total_len = len(df)
  train_size = int(total_len * train_frac)
  dev_size   = int(total_len * dev_frac)
  test_size  = total_len - train_size - dev_size # not used, but for clarity

  train = df.iloc[:train_size]
  dev   = df.iloc[train_size:train_size + dev_size]
  test  = df.iloc[train_size + dev_size:]

  if verbose:
      print(f"Split sizes — train: {len(train)}, dev: {len(dev)}, test: {len(test)}")

  if denoise_fn is not None: # denoise using wavelet denoising
      train = pd.DataFrame({col: denoise_fn(train[col]) for col in keep_cols})

  if scaler_factory is not None:
      scaler = scaler_factory(**(scaler_kwargs or {}))
  else:
      scaler = None


  trainX_untr, trainX, trainY_untr, trainY = create_dataset(train.values, scaler=scaler, look_back=look_back)
  devX_untr,   devX,   devY_untr,   devY   = create_dataset(dev.values,  scaler=scaler, look_back=look_back)
  testX_untr,  testX,  testY_untr,  testY  = create_dataset(test.values, scaler=scaler, look_back=look_back)

  if add_technical_indicators:
      # Predict S1_close using all other columns except S1_close as X, making it multivariate regression with a very large number of variables (the technical indicators)
      y_train = train[col_s1].values
      X_train = train.drop(columns=[col_s1]).values # all input variables

      # do the same for dev and test
      y_dev = dev[col_s1].values
      X_dev = dev.drop(columns=[col_s1]).values
      y_test = test[col_s1].values
      X_test = test.drop(columns=[col_s1]).values

      # apply scaler to all versions of the input
      if scaler is not None:
          X_train = scaler.fit_transform(X_train)
          X_dev = scaler.transform(X_dev)
          X_test = scaler.transform(X_test)

      # recursive least squares multivariate regression, using the function
      beta_t = kalman_filter_regression_multivariate(X_train, y_train, delta=delta)
      forecast_train = np.sum(X_train * beta_t, axis=1)
      forecast_dev   = np.sum(X_dev * beta_t[-len(X_dev):], axis=1)
      forecast_test  = np.sum(X_test * beta_t[-len(X_test):], axis=1)

      if look_back == 1:
          forecast_test_list = [np.array([v]) for v in forecast_test] # this way of calculating MSE is a bit messy and too many steps, but it works in the workflow now, so we'll keep it like this.
          testY_arr = np.array(y_test).reshape(-1, 1)
          testY_list = [np.array([v]) for v in y_test]
          test_mse = acc_metric(testY_list, forecast_test_list)

          # # repeat for dev / validaiton
          forecast_dev_list = [np.array([v]) for v in forecast_dev]
          devY_arr = np.array(y_dev).reshape(-1, 1)
          devY_list = [np.array([v]) for v in y_dev]
          val_mse = acc_metric(devY_list, forecast_dev_list)
      else:
          print("Warning: look_back > 1 not yet implemented. Returning None for mse.")
          test_mse, val_mse = None, None
  else:
    # get beta_t, the Kalman-filtered regression coefficients
    beta_t = kalman_filter_regression(
        kalman_filter_average(pair_data[col_s1],
                              transition_cov=trans_cov_avg,
                              obs_cov=obs_cov_avg),
        kalman_filter_average(pair_data[col_s2],
                              transition_cov=trans_cov_avg,
                              obs_cov=obs_cov_avg),
        delta=delta,
        obs_cov=obs_cov_reg
    )[:, 0]

    kalman_spread = normalise_fn(
        pair_data[col_s1] + pair_data[col_s2] * beta_t)

    forecast_train = kalman_spread[:len(trainX)].to_numpy() # Though the variable `forecast_train` is never directly used as a variable, the data for it WAS used, during kalman filter averaging and regression
    forecast_dev   = kalman_spread[len(trainX):len(trainX) + len(devX)].to_numpy()
    forecast_test = kalman_spread[-len(testX):].to_numpy()

    if look_back == 1:
        yhat_KF_mse = [np.array([v]) for v in forecast_test]

        # Original normalisation: operate directly on the list-of-arrays
        testY_arr  = np.array(testY_untr)               # shape (N,1)
        testY_norm = (testY_arr - testY_arr.mean()) / testY_arr.std()

        # Convert back to list-of-arrays so acc_metric sees the same layout
        testY_norm_list = [row for row in testY_norm]

        test_mse = acc_metric(testY_norm_list, yhat_KF_mse)

        yhat_KF_dev_mse = [np.array([v]) for v in forecast_dev]
        devY_arr = np.array(devY_untr)
        devY_norm = (devY_arr - devY_arr.mean()) / devY_arr.std()
        devY_norm_list = [row for row in devY_norm]
        val_mse = acc_metric(devY_norm_list, yhat_KF_dev_mse)
    else:
        print("Warning: look_back > 1 not yet implemented. Returning None for mse.")
        test_mse = None
        val_mse = None

  ### TRADING ###
  # calculate std_dev_pct using the logic from plot_with_uncertainty. Then put that into two separate functions: calculate_yoy_uncertainty and a version of plot_with_uncertainty that uses calculate_yoy_uncertainty
  # position threshold (2.00-4.00), clearing threshold (0.30-0.70)
  min_position = 2.00
  max_position = 4.00
  min_clearing = 0.30
  max_clearing = 0.70
  position_thresholds = np.linspace(min_position, max_position, num=10)
  clearing_thresholds = np.linspace(min_clearing, max_clearing, num=10)

  test_index_shortened = test.index[:len(testY_untr)]
  forecast_test_shortened_series = pd.Series(forecast_test[:len(testY_untr)], index=test_index_shortened)
  testY_untr_shortened = pd.Series(testY_untr, index=test_index_shortened)
  test_s1_shortened = test['S1_close'].iloc[:len(testY_untr)]
  test_s2_shortened = test['S2_close'].iloc[:len(testY_untr)]

  yoy_mean, yoy_std = calculate_return_uncertainty(test_s1_shortened, test_s2_shortened, forecast_test_shortened_series, position_thresholds=position_thresholds, clearing_thresholds=clearing_thresholds)
  # calculate the strategy returns if we were to feed the groundtruth values to the `trade` func. If the ground truth returns are lower, it seems likely there is something wrong with the `trade` func (but not certain! Probability applies here).
  # forecast_test_shortened = forecast_test[:len(testY_untr)]
  # spread_pred_series = pd.Series(forecast_test_shortened, index=index_shortened)
  index_shortened = test.index[:len(test['Spread_Close'].values[look_back:])]
  spread_gt_series = pd.Series(test['Spread_Close'].values[look_back:], index=index_shortened)
  gt_returns = trade(
      S1 = test['S1_close'].iloc[look_back:],
      S2 = test['S2_close'].iloc[look_back:],
      spread = spread_gt_series,
      window_long = 30,
      window_short = 5,
      position_threshold = 3,
      clearing_threshold = 0.4
  )
  gt_yoy = ((gt_returns[-1] / gt_returns[0])**(365 / len(gt_returns)) - 1)[0]

  if add_technical_indicators:
    current_result_dir = filename_base.replace(".pkl", "_kalman")
  else:
    current_result_dir = filename_base.replace(".pkl", "_kalman_without_ta")  
  result_dir = os.path.join(result_parent_dir, current_result_dir)
  if not os.path.exists(result_dir):
      os.makedirs(result_dir)
  yoy_returns_filename = plot_return_uncertainty(test_s1_shortened, test_s2_shortened, forecast_test_shortened_series, test_index_shortened, look_back, position_thresholds=position_thresholds, clearing_thresholds=clearing_thresholds, verbose=verbose, result_dir=result_dir, filename_base=filename_base)
  predicted_vs_actual_spread_filename = plot_comparison(testY_untr, forecast_test, test.index, workflow_type="Kalman Filter", pair_tup_str=pair_tup_str, verbose=verbose, result_dir=result_dir, filename_base=filename_base)
  plot_filenames = {
      "yoy_returns": yoy_returns_filename,
      "predicted_vs_actual_spread": predicted_vs_actual_spread_filename,
      "train_val_loss": None
  }
  # save results to .txt file
  results_str = f"""
Validation MSE: {val_mse}
Test MSE: {test_mse}
YOY Returns: {yoy_mean * 100:.2f}%
YOY Std: +- {yoy_std * 100:.2f}%
GT Yoy: {gt_yoy * 100:.2f}%
Plot filepath parent dir: {result_parent_dir}
Plot filenames: {plot_filenames}
  """
  with open(os.path.join(result_dir, "results.txt"), "w") as f:
      f.write(results_str)


  if verbose:
    print(results_str)
  output: Dict[str, Any] = dict(
      val_mse=val_mse,
      test_mse=test_mse,
      yoy_mean=yoy_mean,
      yoy_std=yoy_std,
      gt_yoy=gt_yoy,
      result_parent_dir=result_parent_dir,
      plot_filenames=plot_filenames
  )
  if return_datasets:
      output.update(
          dict(train=train, dev=dev, test=test,
                datasets=dict(
                    train=(trainX_untr, trainX, trainY_untr, trainY),
                    dev  =(devX_untr,   devX,   devY_untr,   devY),
                    test =(testX_untr,  testX,  testY_untr,  testY)
                ))
      )
  return output
