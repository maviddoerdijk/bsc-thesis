from typing import Optional, Callable, Dict, Any
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from sklearn.preprocessing import MinMaxScaler

# custom imports
from utils.visualization import plot_train_val_loss, plot_return_uncertainty, plot_comparison
from models.statistical_models import default_normalize
from preprocessing.wavelet_denoising import wav_den
from backtesting.trading_strategy import trade
from backtesting.utils import calculate_return_uncertainty


def execute_transformer_workflow(
  pairs_timeseries: pd.DataFrame,
  target_col: str = "Spread_Close",
  burn_in: int = 30, # we remove the first 30 elements, because the largest window used for technical indicators is
  train_frac: float = 0.90,
  dev_frac: float = 0.05,   # remaining part is test
  look_back: int = 20,
  batch_size: int = 64,
  epochs: int = 400,
  patience: int = 150,
  denoise_fn: Optional[Callable[[pd.Series], np.ndarray]] = wav_den,
  scaler_factory: Callable[..., MinMaxScaler] = MinMaxScaler,
  scaler_kwargs: Optional[Dict[str, Any]] = {"feature_range": (0, 1)},
  normalise_fn: Callable[[pd.Series], pd.Series] = default_normalize,
  return_datasets: bool = False,
  verbose: bool = False,
  add_technical_indicators: bool = True,
  result_parent_dir: str = "data/results",
  filename_base: str = "data_begindate_enddate_hash.pkl",
  pair_tup_str: str = "(?,?)" # Used for showing which tuple was used in plots, example: "(QQQ, SPY)"
):
  if not target_col in pairs_timeseries.columns:
    raise KeyError(f"pairs_timeseries must contain {target_col}")

  # burn the first 30 elements
  pairs_timeseries_burned = pairs_timeseries.iloc[burn_in:].copy()

  total_len = len(pairs_timeseries_burned)
  train_size = int(total_len * train_frac)
  dev_size   = int(total_len * dev_frac)
  test_size  = total_len - train_size - dev_size # not used, but for clarity

  train = pairs_timeseries_burned[:train_size]
  dev   = pairs_timeseries_burned[train_size:train_size+dev_size] # aka validation
  test  = pairs_timeseries_burned[train_size+dev_size:]

  train_multivariate = train.copy()
  dev_multivariate   = dev.copy() # only for completeness
  test_multivariate  = test.copy() # only for completeness

  if verbose:
      print(f"Split sizes — train: {len(train)}, dev: {len(dev)}, test: {len(test)}")

  if denoise_fn is not None: # denoise using wavelet denoising
      train = pd.DataFrame({col: denoise_fn(train[col]) for col in train.columns}) # TODO: unsure whether dev and test should also be denoised?

  x_scaler = scaler_factory(**scaler_kwargs) # important: the scaler learns parameters, so separate objects must be created for x and y
  y_scaler = scaler_factory(**scaler_kwargs)

  if not add_technical_indicators:
      train = train[[target_col]]
      dev = dev[[target_col]]
      test = test[[target_col]]

  # We want a sliding window in our dataset
  # TODO: defining this function should not be part of workflow, but imported from a custom module
  def create_sliding_dataset(mat: np.ndarray,
                            x_scaler: MinMaxScaler,
                            y_scaler: MinMaxScaler,
                            look_back: int = 20):
      """
      X  -> (samples, look_back, features)
      y  -> (samples, 1)   — the next-step Spread_Close (just 1 day in advance)
      """
      X, y = [], []
      for i in range(len(mat) - look_back):
          X.append(mat[i : i + look_back, :]) # window
          y.append(mat[i + look_back, 0]) # value right after the window
      X, y = np.array(X), np.array(y).reshape(-1, 1)

      # scale per feature (fit on the training set once!)
      X_scaled = x_scaler.fit_transform(
          X.reshape(-1, X.shape[-1])
      ).reshape(X.shape)
      y_scaled = y_scaler.fit_transform(y)

      return X, X_scaled, y, y_scaled

  trainX_raw, trainX_scaled, trainY_raw, trainY_scaled = create_sliding_dataset(
      train.values, x_scaler=x_scaler, y_scaler=y_scaler, look_back=look_back) # train_X_scaled.shape: (2219, 20, 34) // [(t - look_back), look_back, features]
  devX_raw,   devX_scaled,   devY_raw,   devY_scaled   = create_sliding_dataset(
      dev.values,  x_scaler=x_scaler, y_scaler=y_scaler, look_back=look_back)
  testX_raw,  testX_scaled,  testY_raw,  testY_scaled  = create_sliding_dataset(
      test.values, x_scaler=x_scaler, y_scaler=y_scaler, look_back=look_back)


  # use pytorch Dataset class
  class SlidingWindowDataset(Dataset):
      def __init__(self, X: np.ndarray, y: np.ndarray):
          #  cast to float32 once to avoid repeated conversions
          self.X = torch.tensor(X, dtype=torch.float32)      # (N, L, F)
          self.y = torch.tensor(y, dtype=torch.float32)      # (N, 1)

      def __len__(self):
          return self.X.shape[0]

      def __getitem__(self, idx):
          return self.X[idx], self.y[idx]                    # each X: (L, F)

  train_ds = SlidingWindowDataset(trainX_scaled, trainY_scaled)
  dev_ds   = SlidingWindowDataset(devX_scaled, devY_scaled)
  test_ds  = SlidingWindowDataset(testX_scaled, testY_scaled)

  train_loader = DataLoader(train_ds, batch_size=batch_size,
                            shuffle=True,  drop_last=True,  num_workers=0)
  dev_loader   = DataLoader(dev_ds,   batch_size=batch_size,
                            shuffle=False, drop_last=False, num_workers=0)
  test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                            shuffle=False, drop_last=False, num_workers=0)

  if verbose:
    print(f"Single tensor shape: {next(iter(train_loader))[0].shape}")   # torch.Size([64, 20, 34]) //  (batch_size, look_back, features)

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
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1,
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

  n_features = trainX_scaled.shape[-1]

  DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
  model  = TimeSeriesTransformerv1(
              n_features=n_features,
              seq_len=look_back,
              d_model=128,
              nhead=8,
              num_layers=4,
              dropout=0.1).to(DEVICE)

  criterion = nn.MSELoss()
  optimizer = AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)

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
          loss  = criterion(preds, y_batch.squeeze(-1))
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

      # print losses in a pretty way
      if epoch % print_per_n == 0:
        print(f"Epoch {epoch:03d} | train MSE {train_loss:.6f} | val MSE {val_loss:.6f}")


      # manual early stopping logic
      if val_loss < best_val - 1e-5: # 1e-5 to not actually make it zero
          best_val = val_loss
          epochs_no_improve = 0
          torch.save(model.state_dict(), "best_transformer.pt")
      else:
          epochs_no_improve += 1
          if epochs_no_improve >= PATIENCE:
              print("Early stopping triggered.")
              break

  # Now, let's run the model on the testset
  # made sure we're in eval mode
  model.eval()

  all_preds, all_targets = [], []
  with torch.no_grad(): # note for myself: torch.no_grad() makes sure that individual weights are not stored in memory, because we would only need to know those during learning, not during inference
      for x_test_batch, y_test_batch in test_loader:
        x_test_batch = x_test_batch.to(DEVICE)
        preds = model(x_test_batch) # make predictions using model
        # transform the preds and targets back to numpy, as these need to be inverted with the scaler, which expects numpy not tensors
        preds = preds.cpu().numpy()
        y_test_batch = y_test_batch.cpu().numpy()
        all_preds.append(preds)
        all_targets.append(y_test_batch)

  # maybe too much explanation here, but y_hat and y_true respectively represent the predicted and ground truth values
  y_hat_scaled = np.concatenate(all_preds).reshape(-1, 1)
  y_true_scaled = np.concatenate(all_targets).reshape(-1, 1)

  y_hat = y_scaler.inverse_transform(y_hat_scaled)
  y_true = y_scaler.inverse_transform(y_true_scaled)

  test_mse = np.mean((y_hat - y_true) ** 2)
  print(f"Test MSE  : {test_mse:.6f}")

  ## Trading
  test_s1_shortened = test_multivariate['S1_close'].iloc[look_back:]
  test_s2_shortened = test_multivariate['S2_close'].iloc[look_back:] # use multivariate versions, so we can still access cols like 'S1_close' and 'S2_close'
  test_index_shortened = test_multivariate.index[look_back:] # officially doesn't really matter whether to use `test_multivariate` or `test`, but do it like this for consistency
  forecast_test_shortened_series = pd.Series(y_hat.squeeze(), index=test_index_shortened)
  gt_test_shortened_series = pd.Series(y_true.squeeze(), index=test_index_shortened)

  spread_gt_series = pd.Series(y_true.squeeze(), index=test_index_shortened)
  gt_returns = trade(
      S1 = test_s1_shortened,
      S2 = test_s2_shortened,
      spread = spread_gt_series,
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


  ## The variables that should be returned, according to what was returned by the `execute_kalman_workflow` func:
  # give same output as was originally the case
  if add_technical_indicators:
    current_result_dir = filename_base.replace(".pkl", "_transformer")
  else:
    current_result_dir = filename_base.replace(".pkl", "_transformer_without_ta")
  result_dir = os.path.join(result_parent_dir, current_result_dir)
  if not os.path.exists(result_dir):
      os.makedirs(result_dir)

  ### Plotting #####
  # 1. Train val loss
  train_val_loss_filename = plot_train_val_loss(train_losses, val_losses, workflow_type="Transformer", pair_tup_str=pair_tup_str, result_dir=result_dir, verbose=verbose, filename_base=filename_base)

  # 2. yoy returns
  yoy_returns_filename = plot_return_uncertainty(test_s1_shortened, test_s2_shortened, forecast_test_shortened_series, test_index_shortened, look_back, position_thresholds=position_thresholds, clearing_thresholds=clearing_thresholds, verbose=verbose, result_dir=result_dir, filename_base=filename_base)

  # 3. predicted vs actual spread plot
  predicted_vs_actual_spread_filename = plot_comparison(gt_test_shortened_series, forecast_test_shortened_series, test_index_shortened, workflow_type="Kalman Filter", pair_tup_str=pair_tup_str, verbose=verbose, result_dir=result_dir, filename_base=filename_base)

  ### Plotting #####
  plot_filenames = {
      "yoy_returns": yoy_returns_filename,
      "predicted_vs_actual_spread": predicted_vs_actual_spread_filename,
      "train_val_loss": train_val_loss_filename
  }
  output: Dict[str, Any] = dict(
      val_mse=val_losses[-1],
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

  if return_datasets:
      output.update(
          dict(train=train, dev=dev, test=test,
                datasets=dict(
                    train=(trainX_raw, trainX_scaled, trainY_raw, trainY_scaled),
                    dev  =(devX_raw,   devX_scaled,   devY_raw,   devY_scaled),
                    test =(testX_raw,  testX_scaled,  testY_raw,  testY_scaled)
                ))
      
      )
  return output
