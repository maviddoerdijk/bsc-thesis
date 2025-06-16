from typing import Optional, Callable, Dict, Any
import os
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR
from torch.optim import AdamW
import random
import math

# custom imports
from backtesting.trading_strategy import get_gt_yoy_returns_test_dev
from backtesting.utils import calculate_return_uncertainty

def get_cosine_schedule_with_warmup_and_min_lr(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    min_lr,
    lr,
    last_epoch=-1
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress)) 
        min_lr_ratio = min_lr / lr
        return max(min_lr_ratio, cosine_decay * (1 - min_lr_ratio) + min_lr_ratio)
    return LambdaLR(optimizer, lr_lambda, last_epoch)

class TimeSeriesTransformerv1(nn.Module):
  """
  This version (v1) uses:
  * learnable positional embeddings (simple, so no RoPE and no sinusoidal)
  * only an encoder (followed by a regression head that transforms from form (seq_len, d_model) into (1), with the output form being the Spread_Close prediction)
  """
  def __init__(
      self,
      n_features: int,
      seq_len: int,
      d_model: int,
      nhead: int,
      num_layers: int,
      dropout: float,
  ):
      super().__init__()
      self.seq_len = seq_len

      # token projection (linear layer)
      self.input_proj = nn.Linear(n_features, d_model)

      # learnable positional embedding  (1, seq_len, d_model)
      self.pos_emb = nn.Parameter(torch.randn(1, seq_len, d_model))

      # encoder (important part)
      enc_layer = nn.TransformerEncoderLayer(
          d_model=d_model,
          nhead=nhead,
          dim_feedforward=d_model * 4,
          dropout=dropout,
          batch_first=True, # keeps (batch, seq, dim)
      )
      self.encoder = nn.TransformerEncoder(enc_layer, num_layers)

      # regression head (mainly helps in getting to the right output format)
      self.head = nn.Sequential(
          nn.Flatten(start_dim=1), # (batch, seq_len*d_model)
          nn.Linear(seq_len * d_model, 128),
          nn.ReLU(),
          nn.Dropout(dropout),
          nn.Linear(128, 1),
      )

  def forward(self, x): # x: (batch, seq_len, n_features)
      x = self.input_proj(x) + self.pos_emb
      x = self.encoder(x) # (batch, seq_len, d_model)
      return self.head(x) # (batch, 1)

def execute_transformer_workflow(
  pairs_timeseries: pd.DataFrame,
  target_col: str = "Spread_Close",
  col_s1: str = "S1_close",
  col_s2: str = "S2_close",
  train_frac: float = 0.90,
  dev_frac: float = 0.05,   # remaining part is test
  seed: int = 3178749, # for reproducibility, my student number
  look_back: int = 20,
  yearly_trading_days: int = 252,
  ## optimized hyperparams: architecture ##
  d_model: int = 256, 
  nhead: int = 8,
  num_layers: int = 4,
  dropout: float = 0.1,
  ## optimized hyperparams: architecture ##
  ## optimized hyperparams: learning algorithm ##
  learning_rate: float = 1e-4,
  min_learning_rate: float = 5e-5,
  warmup_ratio: float = 0.0,
  weight_decay: float = 0.1,
  batch_size: int = 64, 
  adam_beta1: float = 0.9,
  adam_beta2: float = 0.95,
  adam_epsilon: float = 1e-8,
  ## optimized hyperparams: learning algorithm ##
  epochs: int = 400,
  patience: int = 150,
  return_datasets: bool = False,
  verbose: bool = False,
  result_parent_dir: str = "data/results",
  filename_base: str = "data_begindate_enddate_hash.pkl",
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
      
  DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
  if verbose:
    print(f"Using device: {DEVICE}")

  if not target_col in pairs_timeseries.columns:
    raise KeyError(f"pairs_timeseries must contain {target_col}")

  total_len = len(pairs_timeseries)
  train_size = int(total_len * train_frac)
  dev_size   = int(total_len * dev_frac)
  test_size  = total_len - train_size - dev_size # not used, but for clarity

  train_univariate = pairs_timeseries[:train_size][[target_col]]
  dev_univariate = pairs_timeseries[train_size:train_size+dev_size][[target_col]] # aka validation
  test_univariate = pairs_timeseries[train_size+dev_size:][[target_col]]
  
  train_multivariate = pairs_timeseries[:train_size]
  dev_multivariate = pairs_timeseries[train_size:train_size+dev_size] # aka validation
  test_multivariate = pairs_timeseries[train_size+dev_size:]

  if verbose:
      print(f"Split sizes — train: {len(train_univariate)}, dev: {len(dev_univariate)}, test: {len(test_univariate)}")
    
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

  trainX_raw, trainX_scaled, trainY_raw, trainY_scaled, train_mean, train_std = create_sequences_rolling(train_univariate, look_back)
  devX_raw, devX_scaled, devY_raw, devY_scaled, dev_mean, dev_std = create_sequences_rolling(dev_univariate, look_back, train_mean, train_std)
  testX_raw, testX_scaled, testY_raw, testY_scaled, _, _ = create_sequences_rolling(test_univariate, look_back, dev_mean, dev_std)


  # use pytorch Dataset class
  class SlidingWindowDataset(Dataset):
      def __init__(self, X: np.ndarray, y: np.ndarray):
          self.X = X
          self.y = y  # both already casted to torch tensors in create_sequences_rolling

      def __len__(self):
          return self.X.shape[0]

      def __getitem__(self, idx):
          return self.X[idx], self.y[idx]

  train_ds = SlidingWindowDataset(trainX_scaled, trainY_scaled)
  dev_ds   = SlidingWindowDataset(devX_scaled, devY_scaled)
  test_ds  = SlidingWindowDataset(testX_scaled, testY_scaled)

  train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  drop_last=True, num_workers=0) # workers=0 causes it to be processed by main process (cpu)
  dev_loader   = DataLoader(dev_ds, batch_size=batch_size,shuffle=False, drop_last=False, num_workers=0)
  test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=0)
  
  n_features = trainX_scaled.shape[-1]
      
  model = TimeSeriesTransformerv1(
              n_features=n_features,
              seq_len=look_back,
              d_model=d_model,
              nhead=nhead,
              num_layers=num_layers,
              dropout=dropout).to(DEVICE)

  criterion = nn.MSELoss()
  optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=(adam_beta1, adam_beta2), eps=adam_epsilon)
  # learning_rate: float = 1e-4,
  # min_learning_rate: float = 5e-5,
  # warmup_ratio: float = 0.0,
  num_training_steps = len(train_loader) * epochs
  num_warmup_steps = int(warmup_ratio * num_training_steps)
  lr_scheduler = get_cosine_schedule_with_warmup_and_min_lr(
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
    min_lr=min_learning_rate,
    lr=learning_rate
  )

  EPOCHS = epochs
  PATIENCE = patience

  # implement the early stopping logic manually
  best_val = float("inf")
  epochs_no_improve = 0
  print_per_n = 10

  # save train_loss and val_loss to lists for plotting
  train_losses = []
  val_losses = []

  for epoch in range(1, EPOCHS + 1):
      model.train()
      running_loss = 0.0
      for x_batch, y_batch in train_loader:
          x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
          optimizer.zero_grad()
          preds = model(x_batch).squeeze(-1)
          loss = criterion(preds, y_batch.squeeze(-1))
          loss.backward()
          optimizer.step()
          running_loss += loss.item() * x_batch.size(0)
      train_loss = running_loss / len(train_loader.dataset) # epoch loss = running loss / N samples
      train_losses.append(train_loss)

      model.eval()
      running_loss_val = 0.0
      with torch.no_grad():
          for x_batch, y_batch in dev_loader:
              x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
              preds  = model(x_batch).squeeze(-1)
              running_loss_val += criterion(preds, y_batch.squeeze(-1)).item() * x_batch.size(0)
      val_loss = running_loss_val / len(dev_loader.dataset) # again, epoch loss = running loss / N samples
      val_losses.append(val_loss)
      
      # update learning rate scheduler 
      lr_scheduler.step()

      # print losses in a pretty way
      if epoch % print_per_n == 0:
        print(f"Epoch {epoch:03d} | train MSE {train_loss:.6f} | val MSE {val_loss:.6f}")


      # manual early stopping logic
      if val_loss < best_val - 1e-5: # 1e-5 to not actually make it zero
          best_val = val_loss
          epochs_no_improve = 0
          # model would normally be saved here for later use, but the current worfklow does not do this
          # torch.save(model.state_dict(), "best_transformer.pt")
      else:
          epochs_no_improve += 1
          if epochs_no_improve >= PATIENCE:
              print("Early stopping triggered.")
              break

  # Now, let's evaluate the model on the testset
  # made sure we're in eval mode
  model.eval()

  # Helper: Get preds/targets from dataloader, in scaled space
  def get_preds_targets_scaled(dataloader, model, device):
      all_preds = []
      all_targets = []
      model.eval()
      with torch.no_grad():
          for x_batch, y_batch in dataloader:
              x_batch, y_batch = x_batch.to(device), y_batch.to(device)
              preds = model(x_batch).cpu().numpy()
              targets = y_batch.cpu().numpy()
              all_preds.append(preds)
              all_targets.append(targets)
      all_preds = np.concatenate(all_preds).reshape(-1, 1)
      all_targets = np.concatenate(all_targets).reshape(-1, 1)
      return all_preds, all_targets

  ## GETTING MSE's
  # turn train_std and train_mean into numpy arrays, because torch tensors cannot be used in combination with np
  train_mean, train_std = np.array(train_mean), np.array(train_std)
  dev_mean, dev_std = np.array(dev_mean), np.array(dev_std)
  # VAL (DEV)
  val_preds_scaled, val_targets_scaled = get_preds_targets_scaled(dev_loader, model, DEVICE)
  # Inverse-transform to original space
  val_preds_original_scale = val_preds_scaled * train_std + train_mean
  val_targets_original_scale = val_targets_scaled * train_std + train_mean
  val_mse_after_inverse = mean_squared_error(val_targets_original_scale, val_preds_original_scale)
  val_var = np.var(val_targets_original_scale)
  val_nmse = val_mse_after_inverse / val_var # normalized mean squared error

  # TEST
  test_preds_scaled, test_targets_scaled = get_preds_targets_scaled(test_loader, model, DEVICE)
  test_preds_original_scale = test_preds_scaled * dev_std + dev_mean
  test_targets_original_scale = test_targets_scaled * dev_std + dev_mean
  test_mse_after_inverse = mean_squared_error(test_targets_original_scale, test_preds_original_scale)
  test_var = np.var(test_targets_original_scale)
  test_nmse = test_mse_after_inverse / test_var

  # maybe too much explanation here, but y_hat and y_true respectively represent the predicted and ground truth values
  y_hat_scaled = np.concatenate(test_preds_scaled).reshape(-1, 1)
  y_true_scaled = np.concatenate(test_targets_scaled).reshape(-1, 1)

  y_hat = y_hat_scaled * dev_std + dev_mean
  y_true = y_true_scaled * dev_std + dev_mean

  ## Trading
  test_s1_shortened = test_multivariate[col_s1].iloc[look_back:]
  test_s2_shortened = test_multivariate[col_s2].iloc[look_back:] # use multivariate versions, so we can still access cols like 'S1_close' and 'S2_close'
  test_index_shortened = test_multivariate.index[look_back:] # officially doesn't really matter whether to use `test_multivariate` or `test`, but do it like this for consistency
  forecast_test_shortened_series = pd.Series(y_hat.squeeze(), index=test_index_shortened)
  gt_test_shortened_series = pd.Series(y_true.squeeze(), index=test_index_shortened)

  output = get_gt_yoy_returns_test_dev(pairs_timeseries, dev_frac, train_frac, look_back=20, yearly_trading_days=yearly_trading_days)
  gt_yoy, gt_yoy_for_dev_dataset = output['gt_yoy_test'], output['gt_yoy_dev']

  ## Trading: Mean YoY
  min_position = 3.00
  max_position = 3.50
  min_clearing = 0.40
  max_clearing = 0.50
  position_thresholds = np.linspace(min_position, max_position, num=10)
  clearing_thresholds = np.linspace(min_clearing, max_clearing, num=10)
  yoy_mean, yoy_std = calculate_return_uncertainty(test_s1_shortened, test_s2_shortened, forecast_test_shortened_series, position_thresholds=position_thresholds, clearing_thresholds=clearing_thresholds)

  ## The variables that should be returned, according to what was returned by the `execute_kalman_workflow` func:
  # give same output as was originally the case
  current_result_dir = filename_base.replace(".pkl", "_transformer")
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
            test_s1_shortened=test_s1_shortened, 
            test_s2_shortened=test_s2_shortened, 
            forecast_test_shortened_series=forecast_test_shortened_series, 
            gt_test_shortened_series=gt_test_shortened_series
          )
      )
  return output