## data gathering imports
from utils.helpers import _get_train_dev_frac
from preprocessing.filters import step_1_filter_remove_nans, step_2_filter_liquidity
from preprocessing.cointegration import find_cointegrated_pairs
from preprocessing.data_preprocessing import filter_pairs_data
from preprocessing.technical_indicators import combine_pairs_data
## specific caching imports (should be changed in case you want to gather data live)
from data.scraper import load_cached_etf_tickers
from data.data_collection_cache import gather_data_cached, gather_data_cached_using_truncate, gather_pairs_data_cached, save_pairs_data_filtered

## workflow imports
from models.statistical_models import execute_kalman_workflow

## optimize-specific imports
from skopt import gp_minimize # requires: scikit-optimize
from skopt.space import Real
from skopt.utils import use_named_args
import numpy as np
from typing import Callable, Any, List, Dict, Tuple

# extra
import time

def bayesian_optimize_workflow(
    execute_workflow_fn: Callable,
    top_pair_count: int,
    start_year: int,
    min_end_year: int,
    max_end_year: int,
    search_space: List[Real],
    n_calls:int,
    seed: int,
    verbose: bool
) -> Tuple[
    Dict[str, Any], # best_params
    float # best_mean_mse
]:
    global res
    param_names = [dim.name for dim in search_space]
    instrumentIds = load_cached_etf_tickers()

    @use_named_args(search_space)
    def objective(**params):
      total_mse_list = []

      # time series cross validation: go over all periods
      for rolling_end_year in range(min_end_year, max_end_year + 1): # +1 such that end year is actually included!
        startDateStr = f"{start_year}-01-01"
        endDateStr = f"{rolling_end_year}-12-31"
        startDateStrTest = f"{rolling_end_year}-01-01"
        endDateStrTest = endDateStr

        train_frac, dev_frac = _get_train_dev_frac(startDateStr, endDateStr, startDateStrTest, endDateStrTest)

        # when new startDateStr and endDateStr are created, we also need new pairs_data_filtered and data_..._filtered_2
        data = gather_data_cached_using_truncate(startDateStr, endDateStr, instrumentIds, cache_dir='../src/data/cache')
        data_close_filtered_1, data_open_filtered_1, data_high_filtered_1, data_low_filtered_1, data_vol_filtered_1, data_original_format_filtered_1 = step_1_filter_remove_nans(data['close'], data['open'], data['high'], data['low'], data['vol'], data)
        data_close_filtered_2, data_open_filtered_2, data_high_filtered_2, data_low_filtered_2, data_vol_filtered_2, data_original_format_filtered_2 = step_2_filter_liquidity(data_close_filtered_1, data_open_filtered_1, data_high_filtered_1, data_low_filtered_1, data_vol_filtered_1, data_original_format_filtered_1)

        pairs_data_filtered = gather_pairs_data_cached(startDateStr, endDateStr, instrumentIds, cache_dir='../src/data/cache')
        if pairs_data_filtered is None:
          scores, pvalues, pairs = find_cointegrated_pairs(data_original_format_filtered_2)
          pairs_data = {key:value[1]  for (key, value) in pairs.items()}
          pairs_data = sorted(pairs_data.items(), key=lambda x: x[1])
          pairs_data_filtered = filter_pairs_data(pairs_data)
          # if it can not be retreived from cache, make sure it is saved for later
          save_pairs_data_filtered(pairs_data_filtered, startDateStr, endDateStr, instrumentIds, cache_dir='../src/data/cache')

        # for more balanced results: use top x pairs, chosen to be 5 due to stay in within realistic time and resources
        for pair_idx in range(top_pair_count):
          ticker_a, ticker_b = pairs_data_filtered[pair_idx][0][0], pairs_data_filtered[pair_idx][0][1]
          pairs_timeseries_df = combine_pairs_data(data_close_filtered_2, data_open_filtered_2, data_high_filtered_2, data_low_filtered_2, data_vol_filtered_2, ticker_a, ticker_b)
          # Note: approximate time for each workflow:
          # kalman; 2-4 sec
          # transformer; ? sec
          # time-moe: ? sec
          output = execute_workflow_fn(
              pairs_timeseries_df,
              **params,
              verbose=False
          )
          total_mse_list.append(output['test_mse'])
      # get mean_mse across time periods and pairs for the current choice of hyperparameters
      print(f"total_mse_list: {total_mse_list}")
      print(f"mean mse: {np.mean(total_mse_list)}")
      mean_mse = np.mean(total_mse_list)
      return mean_mse

    # gather results
    res = gp_minimize(
        func=objective,
        dimensions=search_space,
        n_calls=n_calls,
        n_random_starts=10,
        random_state=seed,
        verbose=verbose
    )
    return res