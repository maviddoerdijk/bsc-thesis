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
from external.time_moe_repo.training_wrapper import train_time_moe
from backtesting.trading_strategy import get_gt_yoy_returns_test_dev
from backtesting.utils import calculate_return_uncertainty

## semi-custom
from external.time_moe_repo.time_moe.models.modeling_time_moe import TimeMoeForPrediction

def execute_timemoe_workflow(
  pairs_timeseries: pd.DataFrame,
  target_col: str = "Spread_Close",
  col_s1: str = "S1_close",
  col_s2: str = "S2_close",
  train_frac: float = 0.90,
  dev_frac: float = 0.05,   # remaining part is test
  seed: int = 3178749, # for reproducibility, my student number
  look_back: int = 20,
  yearly_trading_days: int = 252,
  ## optimized hyperparams ##
  learning_rate=1e-4,
  min_learning_rate=5e-5,
  warmup_ratio=0.0,
  weight_decay=0.1,
  global_batch_size=64, # (just the batch size) other option would be micro_batch_size, which sets batch size per device
  adam_beta1=0.9,
  adam_beta2=0.95,
  adam_epsilon=1e-8,
  ## optimized hyperparams
  return_datasets: bool = False,
  batch_size: int = 8, # TODO: go over which batch size should be used where! (training vs test inference)
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
  
  total_len = len(pairs_timeseries)
  train_size = int(total_len * train_frac)
  dev_size   = int(total_len * dev_frac)
  test_size  = total_len - train_size - dev_size # not used, but for clarity

  pairs_timeseries_univariate = pairs_timeseries[target_col]

  train_univariate = pairs_timeseries_univariate[:train_size]
  dev_univariate = pairs_timeseries_univariate[train_size:train_size+dev_size] # aka validation
  test_univariate = pairs_timeseries_univariate[train_size+dev_size:]

  train_multivariate = pairs_timeseries.iloc[:train_size]
  dev_multivariate = pairs_timeseries.iloc[train_size:train_size+dev_size]
  test_multivariate = pairs_timeseries.iloc[train_size+dev_size:]

  if verbose:
      print(f"Split sizes — train: {len(train_univariate)}, dev: {len(dev_univariate)}, test: {len(test_univariate)}")

  DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
  if verbose:
    print(f"Using device: {DEVICE}")

  def create_sequences(series, mean=None, std=None):
      # series: pd.Series
      X_raw = torch.tensor(series.values, dtype=torch.float32) # note: using .values loses index
      if mean is None:
        # only compute mean if not given
        mean = torch.tensor(np.array(series.mean()), dtype=torch.float32)
      if std is None:
        std = torch.tensor(np.array(series.std()), dtype=torch.float32)
      X_scaled = (X_raw - mean) / (std + 1e-8)
      return X_raw, X_scaled, mean, std

  def create_sequences_rolling(series, look_back, mean=None, std=None):
      X = []
      y = []
      for i in range(len(series) - look_back):
          seq = series.iloc[i:i+look_back].values
          target = series.iloc[i+look_back]
          X.append(seq)
          y.append(target) 

      X = torch.tensor(np.array(X), dtype=torch.float32)
      y = torch.tensor(np.array(y), dtype=torch.float32)
      
      # z-score normalization
      if mean is None:
        mean = torch.tensor(np.array(series.mean()), dtype=torch.float32)
      if std is None:
        std = torch.tensor(np.array(series.std()), dtype=torch.float32)
      X_scaled = (X - mean) / (std + 1e-8)
      # For y, broadcast mean/std to match shape
      y_scaled = (y - mean) / (std + 1e-8) 
      return X, X_scaled, y, y_scaled, mean, std # rolling X (torch tensor), rolling X (torch tensor), torch series, scaled torch series, float, float   

  train_raw, train_scaled, train_mean, train_std = create_sequences(train_univariate) 
  dev_raw, dev_scaled, dev_mean, dev_std = create_sequences(dev_univariate, train_mean, train_std)
  test_raw, test_scaled, _, _ = create_sequences(test_univariate, dev_mean, dev_std)  

  ## use rolling sequences not for training, but still for inferencing dev and test ##
  trainX_raw, trainX_scaled, trainY_raw, trainY_scaled, train_mean, train_std = create_sequences_rolling(train_univariate, look_back)
  devX_raw_rolling, devX_scaled_rolling, devY_raw_rolling, devY_scaled_rolling, dev_mean, dev_std = create_sequences_rolling(dev_univariate, look_back, train_mean, train_std)
  testX_raw_rolling, testX_scaled_rolling, testY_raw_rolling, testY_scaled_rolling, _, _ = create_sequences_rolling(test_univariate, look_back, dev_mean, dev_std) # Note: dev_mean and test_mean may never be used; preventing data leakage

  dev_ds_rolling = TensorDataset(devX_scaled_rolling, devY_scaled_rolling) # goal of TensorDataset class: loading and processing dataset lazily
  test_ds_rolling = TensorDataset(testX_scaled_rolling, testY_scaled_rolling)

  batch_size = int(batch_size)
  dev_loader_rolling = DataLoader(dev_ds_rolling, batch_size=batch_size, shuffle=False)
  test_loader_rolling = DataLoader(test_ds_rolling, batch_size=batch_size, shuffle=False)
  ## use rolling sequences not for training, but still for inferencing dev and test ##
    
  if load_finetuned:
    ## Training (only train in the case where we actually also want to load finetuned :D )
    # save contents of trainX_scaled to jsonl using _get_filename {"sequence": [1.7994326779272853, 2.554412431241829,
    filename_jsonl = filename_base.replace(".pkl", ".jsonl")
    filepath_parent = os.path.join("data", "datasets")
    os.makedirs(filepath_parent, exist_ok=True)
    filepath_jsonl = os.path.join(filepath_parent, filename_jsonl)
    with open(filepath_jsonl, "w") as f: # Train scaled (improves results according to paper, and empirical tests have also shown this)
        json_line = json.dumps({"sequence": train_scaled.tolist()})
        f.write(json_line + "\n")

    train_time_moe(
        data_path=filepath_jsonl,
        dataloader_num_workers=2,
        ## hyperparams ##
        learning_rate=learning_rate,
        min_learning_rate=min_learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        global_batch_size=global_batch_size, # (just the batch size) other option would be micro_batch_size, which sets batch size per device
        adam_beta1=adam_beta1,
        adam_beta2=adam_beta2,
        adam_epsilon=adam_epsilon
        ## hyperparams ##
    ) # after this, model is saved to logs/time_moe as model.safetensors (400+ MB)
    model_dir = "logs/time_moe"
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model = TimeMoeForPrediction.from_pretrained(model_dir, config=config, torch_dtype=torch.float32)
    model.eval()
  else:
    model = AutoModelForCausalLM.from_pretrained(
        'Maple728/TimeMoE-50M',
        trust_remote_code=True,
    )

  prediction_length = 1

  # forecast in batches from dev dataset
  all_predictions = []
  for i, batch in enumerate(test_loader_rolling):
    inputs = batch[0] # is devX_scaled, for now [1] will return error, later [1] will return devY_scaled :D

    # yvals = batch[1]
    # means = batch[2]
    # stds = batch[3]

    output = model.generate(inputs, max_new_tokens=prediction_length)  # shape is [batch_size, look_back + prediction_length]
    normed_predictions = output[:, -prediction_length:]

    # from returned test_mean and test_std, slice the appropriate slices from the series
    input_size_current = inputs.size()
    batch_size_current = input_size_current[0]
    
    preds = normed_predictions * dev_std + dev_mean
    all_predictions.append(preds)

  # Concatenate all predictions
  predictions = torch.cat(all_predictions, dim=0)
  predictions = predictions.squeeze(-1)
  predictions = predictions.detach().numpy()

  # Also get dev/val predictions
  dev_predictions = []
  for i, batch in enumerate(dev_loader_rolling):
    inputs = batch[0]

    output = model.generate(inputs, max_new_tokens=prediction_length)  # shape is [batch_size, look_back + prediction_length]
    normed_predictions = output[:, -prediction_length:]
    input_size_current = inputs.size()
    batch_size_current = input_size_current[0]

    preds = normed_predictions * train_std + train_mean
    dev_predictions.append(preds)
  dev_predictions = torch.cat(dev_predictions, dim=0)
  dev_predictions = dev_predictions.squeeze(-1)
  dev_predictions = dev_predictions.detach().numpy()

  ## Trading
  test_s1_shortened = test_multivariate[col_s1].iloc[look_back:]
  test_s2_shortened = test_multivariate[col_s2].iloc[look_back:] # use multivariate versions, so we can still access cols like 'S1_close' and 'S2_close'
  test_index_shortened = test_multivariate.index[look_back:] # officially doesn't really matter whether to use `test_multivariate` or `test`, but do it like this for consistency
  forecast_test_shortened_series = pd.Series(predictions, index=test_index_shortened)
  gt_test_shortened_series = pd.Series(test_raw.numpy()[look_back:], index=test_index_shortened)

  output = get_gt_yoy_returns_test_dev(pairs_timeseries, dev_frac, train_frac, look_back=20, yearly_trading_days=yearly_trading_days)
  gt_yoy, gt_yoy_for_dev_dataset = output['gt_yoy_test'], output['gt_yoy_dev']
  
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

  dev_mse = mean_squared_error(dev_raw.numpy()[look_back:], dev_predictions)
  test_mse = mean_squared_error(test_raw.numpy()[look_back:], predictions)
  dev_variance = dev_raw.numpy()[look_back:].var()
  dev_nmse = dev_mse / dev_variance if dev_variance != 0 else float('inf')
  test_variance = test_raw.numpy()[look_back:].var()
  test_nmse = test_mse / test_variance if test_variance != 0 else float('inf')

  output: Dict[str, Any] = dict(
      val_mse=dev_nmse,
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
            test_s1_shortened=test_s1_shortened, 
            test_s2_shortened=test_s2_shortened, 
            forecast_test_shortened_series=forecast_test_shortened_series, 
            gt_test_shortened_series=gt_test_shortened_series
          )
      )
  return output