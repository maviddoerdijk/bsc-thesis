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
        train = pd.DataFrame({col: denoise_fn(train[col]) for col in keep_cols})
        
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
            dataY.append(mat[(i + 1):(i + 1 + look_back), 0])
        return (dataX,
                scaler_x.fit_transform(dataX),
                dataY,
                scaler_y.fit_transform(dataY))
        
    trainX_untr, trainX, trainY_untr, trainY = create_dataset(train.values)
    devX_untr,   devX,   devY_untr,   devY   = create_dataset(dev.values)
    testX_untr,  testX,  testY_untr,  testY  = create_dataset(test.values)

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
        yhat_KF_mse = [np.array([v]) for v in forecast]

        # Original normalisation: operate directly on the list-of-arrays
        testY_arr  = np.array(testY_untr)               # shape (N,1)
        testY_norm = (testY_arr - testY_arr.mean()) / testY_arr.std()

        # Convert back to list-of-arrays so acc_metric sees the same layout
        testY_norm_list = [row for row in testY_norm]

        mse = acc_metric(testY_norm_list, yhat_KF_mse)
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
                     train=(trainX_untr, trainX, trainY_untr, trainY),
                     dev  =(devX_untr,   devX,   devY_untr,   devY),
                     test =(testX_untr,  testX,  testY_untr,  testY)
                 ))
        )

    return output


# # ORIGINAL ##
# def execute_kalman_workflow(pair_data):
#     # --- helper functions ---
#     def wav_den(ts_orig):
#         ca, cd = pywt.dwt(ts_orig, 'db8')
#         cat = pywt.threshold(ca, np.std(ca)/8, mode='soft')
#         cdt = pywt.threshold(cd, np.std(cd)/8, mode='soft')
#         ts_rec = pywt.idwt(cat, cdt, 'db8')
#         return ts_rec[1:]

#     def normalize(series):
#         return (series - np.mean(series)) / np.std(series)

#     def acc_metric(true_value, predicted_value):
#         acc_met = 0.0
#         m = len(true_value)
#         for i in range(m):
#             acc_met += mean_squared_error(true_value[i], predicted_value[i])
#         return np.sqrt(acc_met / m)

#     def KalmanFilterAverage(x):
#         kf = KalmanFilter(transition_matrices=[1],
#                           observation_matrices=[1],
#                           initial_state_mean=0,
#                           initial_state_covariance=1,
#                           observation_covariance=1,
#                           transition_covariance=0.01)
#         state_means, _ = kf.filter(x.values)
#         return pd.Series(state_means.flatten(), index=x.index)

#     def KalmanFilterRegression(x, y):
#         delta = 1e-3
#         trans_cov = delta / (1 - delta) * np.eye(2)
#         obs_mat = np.expand_dims(np.vstack([[x], [np.ones(len(x))]]).T, axis=1)
#         kf = KalmanFilter(n_dim_obs=1, n_dim_state=2,
#                           initial_state_mean=[0, 0],
#                           initial_state_covariance=np.ones((2, 2)),
#                           transition_matrices=np.eye(2),
#                           observation_matrices=obs_mat,
#                           observation_covariance=2,
#                           transition_covariance=trans_cov)
#         state_means, _ = kf.filter(y.values)
#         return state_means

#     # --- workflow ---

#     look_back = 1
#     cols = [col for col in pair_data.columns if col != 'date']  # Exclude non-feature columns if any

#     lstm_pair_data = pair_data[cols].iloc[30:]
#     train_size = int(len(lstm_pair_data) * 0.9)
#     dev_size = int((len(lstm_pair_data) - train_size) * 0.5) - 30
#     test_size = len(lstm_pair_data) - train_size - dev_size

#     train = lstm_pair_data.iloc[:train_size]
#     dev = lstm_pair_data.iloc[train_size:train_size + dev_size]
#     test = lstm_pair_data.iloc[train_size + dev_size:]

#     train_den = pd.DataFrame({col: wav_den(train[col]) for col in cols})

#     scaler = MinMaxScaler(feature_range=(0, 1))

#     def create_dataset(dataset):
#         dataX, dataY = [], []
#         for i in range(len(dataset) - look_back):
#             dataX.append(dataset[i, :])
#             dataY.append(dataset[(i+1):(i+1+look_back), 0])
#         return dataX, scaler.fit_transform(dataX), dataY, scaler.fit_transform(dataY)

#     trainX_untr, trainX, trainY_untr, trainY = create_dataset(train_den.values)
#     devX_untr, devX, devY_untr, devY = create_dataset(dev.values)
#     testX_untr, testX, testY_untr, testY = create_dataset(test.values)

#     # Kalman
#     state_means = - KalmanFilterRegression(
#         KalmanFilterAverage(pair_data['S1_close']),
#         KalmanFilterAverage(pair_data['S2_close'])
#     )[:, 0]
#     results = normalize(pair_data['S1_close'] + (pair_data['S2_close'] * state_means))
#     forecast = results[-len(testX):].values

#     if look_back == 1:
#         yhat_KF_mse = [np.array([val]) for val in forecast]
#         mse = acc_metric(normalize(testY_untr), yhat_KF_mse)
#     else:
#         mse = None

#     # Plotting
#     plt.figure(figsize=(15, 7))
#     normalize(pair_data['Spread_Close']).plot(label='Spread z-score')
#     results.plot(label='Kalman_Predicted_Spread')
#     plt.axhline(normalize(pair_data['Spread_Close']).mean(), color='black')
#     plt.axhline(1.0, color='red', linestyle='--')
#     plt.axhline(-1.0, color='green', linestyle='--')
#     plt.legend()
#     plt.title('Kalman Filter Prediction vs Ground Truth Spread')
#     plt.show()

#     return {
#         'mse': mse,
#         'forecast': forecast,
#         'state_means': state_means,
#         'kalman_result': results
#     }