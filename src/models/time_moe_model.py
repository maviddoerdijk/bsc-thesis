import pandas as pd
import numpy as np
import os
import json
from typing import Dict, Any
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error
from transformers import AutoModelForCausalLM, AutoConfig
import random

# custom imports
from utils.visualization import plot_return_uncertainty, plot_comparison
from external.time_moe_repo.training_wrapper import train_time_moe
from backtesting.trading_strategy import trade
from backtesting.utils import calculate_return_uncertainty

## semi-custom
from external.time_moe_repo.time_moe.models.modeling_time_moe import TimeMoeForPrediction



def execute_timemoe_workflow(
  pairs_timeseries: pd.DataFrame,
  target_col: str = "Spread_Close",
  burn_in: int = 30, # we remove the first 30 elements, because the largest window used for technical indicators is
  train_frac: float = 0.90,
  dev_frac: float = 0.05,   # remaining part is test
  seed: int = 3178749, # for reproducibility, my student number
  look_back: int = 20,
  batch_size: int = 8,
  verbose: bool = True,
  load_finetuned = True,
  result_parent_dir: str = "data/results",
  filename_base: str = "data_begindate_enddate_hash.pkl",
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
  
  if not target_col in pairs_timeseries.columns:
    raise KeyError(f"pairs_timeseries must contain {target_col}")
  FLASH_ATTN = False

  # burn the first 30 elements
  pairs_timeseries_burned = pairs_timeseries.iloc[burn_in:].copy()

  total_len = len(pairs_timeseries_burned)
  train_size = int(total_len * train_frac)
  dev_size   = int(total_len * dev_frac)
  test_size  = total_len - train_size - dev_size # not used, but for clarity

  # Standard version of the Time-MoE model can only take in univariate time series. Therefore, we will train only on the target_col
  # TODO: Convert to using multivariate again, a certain type of "multivariate" processing is possible according to the original time-moe paper, but not the version we would want to use. It is not possible to use many different features to enhance the prediction of the target column
  pairs_timeseries_burned_univariate = pairs_timeseries_burned[target_col]

  train = pairs_timeseries_burned_univariate[:train_size]
  dev   = pairs_timeseries_burned_univariate[train_size:train_size+dev_size] # aka validation
  test  = pairs_timeseries_burned_univariate[train_size+dev_size:]

  train_multivariate = pairs_timeseries_burned.iloc[:train_size]
  dev_multivariate = pairs_timeseries_burned.iloc[train_size:train_size+dev_size]
  test_multivariate = pairs_timeseries_burned.iloc[train_size+dev_size:]


  if verbose:
      print(f"Split sizes — train: {len(train)}, dev: {len(dev)}, test: {len(test)}")

  # def create_sequences(series, look_back):
  #     X_raw = series[:batch_size * look_back].to_numpy() # .reshape(batch_size, look_back)
  #     X_raw = torch.tensor(X_raw, dtype=torch.float32)

  #     # normalize devX_raw
  #     mean, std = devX_raw.mean(dim=-1, keepdim=True), devX_raw.std(dim=-1, keepdim=True)
  #     X_scaled = (devX_raw - mean) / std
  #     return X_raw, X_scaled, None, None, mean, std
  DEVICE = "cpu" #  "cuda" if torch.cuda.is_available() else "cpu"


  def create_sequences_rolling(series, look_back):
      X = []
      y = []
      for i in range(len(series) - look_back):
          seq = series.iloc[i:i+look_back].values
          target = series.iloc[i+look_back]
          X.append(seq)
          y.append(target)

      X = torch.tensor(X, dtype=torch.float32)
      X = X.to(DEVICE)

      # normalize
      mean = X.mean(dim=-1, keepdim=True)
      std = X.std(dim=-1, keepdim=True)
      X_scaled = (X - mean) / (std + 1e-8)

      y = torch.tensor(y, dtype=torch.float32)
      y = y.to(DEVICE)
      return X, X_scaled, y, None, mean, std

  devX_raw, devX_scaled, devY_raw, devY_scaled, dev_mean, dev_std = create_sequences_rolling(dev, look_back)
  trainX_raw, trainX_scaled, trainY_raw, trainY_scaled, train_mean, train_std = create_sequences_rolling(train, look_back)
  testX_raw, testX_scaled, testY_raw, testY_scaled, test_mean, test_std = create_sequences_rolling(test, look_back)
  if verbose:
    print(f"devX_raw Shape: {devX_raw.shape}") # entire devX_raw has that shape before dataset and dev_loader logic

  dev_ds = TensorDataset(devX_scaled, devY_raw) # goal of TensorDataset class: loading and processing dataset lazily
  train_ds = TensorDataset(trainX_raw, trainY_raw)
  test_ds = TensorDataset(testX_raw, testY_raw)

  dev_loader = DataLoader(dev_ds, batch_size=batch_size, shuffle=False) # DataLoader takes care of shuffling/sampling/weigthed sampling, batching, using multiprocessing to load the data, use pinned memory etc. (source; https://discuss.pytorch.org/t/what-do-tensordataset-and-dataloader-do/107017)
  train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
  test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

  if verbose:
    print(f"dev_loader tensor Shape: {next(iter(dev_loader))[0].shape}, with a total of {len(dev_loader)} batches") # a single tensor in dev_loader now has shape [batch_size, look_back] as expected

  if load_finetuned:
    ## Training (only train in the case where we actually also want to load finetuned :D )
    # save contents of trainX_scaled to jsonl using _get_filename {"sequence": [1.7994326779272853, 2.554412431241829,
    filename_jsonl = filename_base.replace(".pkl", ".jsonl")
    filepath_parent = os.path.join("data", "datasets")
    os.makedirs(filepath_parent, exist_ok=True)
    filepath_jsonl = os.path.join(filepath_parent, filename_jsonl)
    # first method: try to train it on unnormalized sequence
    with open(filepath_jsonl, "w") as f:
        json_line = json.dumps({"sequence": train.to_list()})
        f.write(json_line + "\n")

    train_time_moe(
        data_path=filepath_jsonl,
        dataloader_num_workers=2
    ) # after this, model is saved to logs/time_moe as model.safetensors (400+ MB)
    model_dir = "logs/time_moe"
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model = TimeMoeForPrediction.from_pretrained(model_dir, config=config, torch_dtype=torch.float32)
    model.eval()
  else:
    model = AutoModelForCausalLM.from_pretrained(
        'Maple728/TimeMoE-50M',
        device_map=DEVICE,
        trust_remote_code=True,
    )
    if FLASH_ATTN: # if FLASH_ATTN, we assume the flash-attention module is installed, and adapt the model to use that
      model = AutoModelForCausalLM.from_pretrained('Maple728/TimeMoE-50M', device_map="auto", attn_implementation='flash_attention_2', trust_remote_code=True)

  prediction_length = 1 # TODO: rather than hardcoding prediction length, make a strategy where we can pick and choose different prediction lengths, and see what is affected by this (returns, std dev, ..)

  # forecast in batches from dev dataset
  all_predictions = []
  for i, batch in enumerate(test_loader):
    inputs = batch[0] # is devX_scaled, for now [1] will return error, later [1] will return devY_scaled :D

    yvals = batch[1]
    # means = batch[2]
    # stds = batch[3]

    output = model.generate(inputs, max_new_tokens=prediction_length)  # shape is [batch_size, look_back + prediction_length]
    normed_predictions = output[:, -prediction_length:]

    # from returned test_mean and test_std, slice the appropriate slices from the series
    input_size_current = inputs.size()
    batch_size_current = input_size_current[0]
    local_means = test_mean[batch_size*i : batch_size*i + batch_size_current]
    local_stds = test_std[batch_size*i : batch_size*i + batch_size_current]

    preds = normed_predictions * local_stds + local_means
    all_predictions.append(preds)

  # Concatenate all predictions
  predictions = torch.cat(all_predictions, dim=0)
  predictions = predictions.squeeze(-1)
  predictions = predictions.detach().numpy()

  # Also get dev/val predictions
  dev_predictions = []
  for i, batch in enumerate(dev_loader):
    inputs = batch[0]

    output = model.generate(inputs, max_new_tokens=prediction_length)  # shape is [batch_size, look_back + prediction_length]
    normed_predictions = output[:, -prediction_length:]
    input_size_current = inputs.size()
    batch_size_current = input_size_current[0]
    # get dev rather than test
    local_means = dev_mean[batch_size*i : batch_size*i + batch_size_current]
    local_stds = dev_std[batch_size*i : batch_size*i + batch_size_current]

    preds = normed_predictions * local_stds + local_means
    dev_predictions.append(preds)
  dev_predictions = torch.cat(dev_predictions, dim=0)
  dev_predictions = dev_predictions.squeeze(-1)
  dev_predictions = dev_predictions.detach().numpy()

  ## Trading
  test_s1_shortened = test_multivariate['S1_close'].iloc[look_back:]
  test_s2_shortened = test_multivariate['S2_close'].iloc[look_back:] # use multivariate versions, so we can still access cols like 'S1_close' and 'S2_close'
  test_index_shortened = test_multivariate.index[look_back:] # officially doesn't really matter whether to use `test_multivariate` or `test`, but do it like this for consistency
  forecast_test_shortened_series = pd.Series(predictions, index=test_index_shortened)
  gt_test_shortened_series = pd.Series(testY_raw.numpy(), index=test_index_shortened)

  gt_returns = trade(
      S1 = test_s1_shortened,
      S2 = test_s2_shortened,
      spread = gt_test_shortened_series,
      window_long = 30,
      window_short = 5,
      position_threshold = 1.0,
      clearing_threshold = 0.5
  )
  gt_yoy = ((gt_returns[-1] / gt_returns[0])**(365 / len(gt_returns)) - 1)

  ## Trading: Mean YoY
  min_position = 2.00
  max_position = 4.00
  min_clearing = 0.30
  max_clearing = 0.70
  position_thresholds = np.linspace(min_position, max_position, num=10)
  clearing_thresholds = np.linspace(min_clearing, max_clearing, num=10)
  yoy_mean, yoy_std = calculate_return_uncertainty(test_s1_shortened, test_s2_shortened, forecast_test_shortened_series, position_thresholds=position_thresholds, clearing_thresholds=clearing_thresholds)

  if load_finetuned:
    current_result_dir = filename_base.replace(".pkl", "_timemoe")
  else:
    current_result_dir = filename_base.replace(".pkl", "_timemoe_only_pretrained")
  result_dir = os.path.join(result_parent_dir, current_result_dir)
  if not os.path.exists(result_dir):
      os.makedirs(result_dir)

  ### Plotting #####
  # (no train/val loss plot, as Time-MoE repo did not support plots or even supplying val losses during training)
  train_val_loss_filename = None

  # 1. yoy returns
  yoy_returns_filename = plot_return_uncertainty(test_s1_shortened, test_s2_shortened, forecast_test_shortened_series, test_index_shortened, look_back, position_thresholds=position_thresholds, clearing_thresholds=clearing_thresholds, verbose=verbose, result_dir=result_dir, filename_base=filename_base)

  # 2. predicted vs actual spread plot
  predicted_vs_actual_spread_filename = plot_comparison(gt_test_shortened_series, forecast_test_shortened_series, test_index_shortened, workflow_type="Time-MoE", pair_tup_str=pair_tup_str, verbose=verbose, result_dir=result_dir, filename_base=filename_base)

  ### Plotting #####

  dev_mse = mean_squared_error(devY_raw.numpy(), dev_predictions)
  test_mse = mean_squared_error(testY_raw.numpy(), predictions)
  plot_filenames = {
      "yoy_returns": yoy_returns_filename,
      "predicted_vs_actual_spread": predicted_vs_actual_spread_filename,
      "train_val_loss": train_val_loss_filename
  }

  output: Dict[str, Any] = dict(
      val_mse=dev_mse,
      test_mse=test_mse,
      yoy_mean=yoy_mean,
      yoy_std=yoy_std,
      gt_yoy=gt_yoy,
      result_parent_dir=result_parent_dir,
      plot_filenames=plot_filenames
  )

  results_str = f"""
  Validation MSE: {output['val_mse']}
  Test MSE: {output['test_mse']}
  YOY Returns: {output['yoy_mean'] * 100:.2f}%
  YOY Std: +- {output['yoy_std'] * 100:.2f}%
  GT Yoy: {output['gt_yoy'] * 100:.2f}%
  Plot filepath parent dir: {output['result_parent_dir']}
  Plot filenames: {output['plot_filenames']}
  """

  with open(os.path.join(result_dir, "results.txt"), "w") as f:
      f.write(results_str)
  if verbose:
    print(results_str)
  return output