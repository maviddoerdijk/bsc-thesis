import pywt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from pykalman import KalmanFilter
import pandas as pd
from matplotlib import pyplot as plt
from preprocessing.wavelet_denoising import wav_den
from typing import Callable, Dict, Any, Optional, Tuple, Sequence, Union

def default_normalize(series: pd.Series) -> pd.Series:
    # z-score normalization
    return (series - series.mean()) / series.std(ddof=0)

def rmse_metric(y_true: Sequence[np.ndarray],
                y_pred: Sequence[np.ndarray]) -> float:
    # Currently using RMSE
    mse = np.mean([mean_squared_error(yt, yp) for yt, yp in zip(y_true, y_pred)])
    return np.sqrt(mse)

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

def execute_kalman_workflow(
    pair_data: pd.DataFrame,
    *,
    col_s1: str = "S1_close",
    col_s2: str = "S2_close",
    burn_in: int = 30,
    train_frac: float = 0.90,
    dev_frac: float = 0.05,   # remaining part is test
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
    verbose: bool = False
) -> Dict[str, Any]:
    """
    End-to-end Kalman-filter workflow for a cointegrated pair.

    Parameters
    ----------
    pair_data : pd.DataFrame
        Must contain at least `col_s1` and `col_s2`.
    burn_in : int
        How many initial rows to discard (kept at 30 for parity with legacy code).
    train_frac, dev_frac : float
        Fractions of the **post-burn-in** data assigned to training
        and development sets.  The remainder is the test set.
    look_back : int
        Sequence length fed to the (future) supervised model.  Currently
        only `1` yields a valid RMSE, but you can still create the
        sliced datasets for consistency.
    denoise_fn : Callable or None
        Function applied **column-wise** to the training frame only.
        Set to `None` to skip wavelet denoising.
    scaler_factory : Callable
        An object that produces a *fitted* scaler.  Defaults to `MinMaxScaler`.
    scaler_kwargs : dict or None
        Extra kwargs forwarded to the scaler constructor.
    normalise_fn : Callable
        How to z-score the spread before scoring / plotting.
    return_datasets : bool
        If `True`, the dict includes the `(train, dev, test)` frames and the
        numpy datasets created by `create_dataset()`.
    """
    
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
        train = pd.DataFrame({c: denoise_fn(train[c]) for c in keep_cols},
                             index=train.index)
        
    scaler_x = scaler_factory(**scaler_kwargs)
    scaler_y = scaler_factory(**scaler_kwargs)
    
    def create_dataset(mat: np.ndarray
                       ) -> Tuple[np.ndarray, np.ndarray,
                                  np.ndarray, np.ndarray]:
        """
        Return  (raw_X, scaled_X, raw_Y, scaled_Y),
        where Y is a 1-step shift of X[:,0].
        """
        dataX, dataY = [], []
        for i in range(len(mat) - look_back):
            dataX.append(mat[i, :])
            dataY.append(mat[i + 1 : i + 1 + look_back, 0])
        raw_X = np.asarray(dataX)
        raw_Y = np.asarray(dataY)
        return (raw_X,
                scaler_x.fit_transform(raw_X),
                raw_Y,
                scaler_y.fit_transform(raw_Y))
        
    trainX_raw, trainX, trainY_raw, trainY = create_dataset(train.values)
    devX_raw,   devX,   devY_raw,   devY   = create_dataset(dev.values)
    testX_raw,  testX,  testY_raw,  testY  = create_dataset(test.values)

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

    forecast = kalman_spread[-len(testX):].to_numpy()


    if look_back == 1:
        yhat_mse_input = [np.array([v]) for v in forecast]
        mse = rmse_metric(normalise_fn(pd.Series(testY_raw.flatten())),
                          yhat_mse_input)
    else:
        # TODO: implement mse for different look_back values
        print("Warning: look_back > 1 not yet implemented. Returning None for mse.")
        mse = None
    
    # give same output as was originally the case
    output: Dict[str, Any] = dict(
        mse=mse,
        forecast=forecast,
        state_means=beta_t,
        kalman_result=kalman_spread
    )

    if return_datasets:
        output.update(
            dict(train=train, dev=dev, test=test,
                 datasets=dict(
                     train=(trainX_raw, trainX, trainY_raw, trainY),
                     dev  =(devX_raw,   devX,   devY_raw,   devY),
                     test =(testX_raw,  testX,  testY_raw,  testY)
                 ))
        )

    return output