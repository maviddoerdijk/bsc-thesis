import pickle
import hashlib
import os

def _tickers_to_hash(ticker_list):
    # return a short deterministic(!!) hash string 
    tickers_str = ','.join(sorted(ticker_list))  # important to sort for consistency
    return hashlib.md5(tickers_str.encode('utf-8')).hexdigest()[:8]  # short hash, doesn't need to be unnecessarily long

def _get_filename(startDateStr, endDateStr, instrumentIds):
    """
    Get the filename for the cached data based on the input parameters.
    """
    tickers_hash = _tickers_to_hash(instrumentIds)
    filename = f"data_{startDateStr.replace('-', '_')}_{endDateStr.replace('-', '_')}_{tickers_hash}.pkl"
    return filename

def gather_data_cached(startDateStr, endDateStr, instrumentIds, cache_dir='.'):
    """
    Load the data from a cached file, given arguments matching gather_data.
    """
    filename = _get_filename(startDateStr, endDateStr, instrumentIds)
    filepath = os.path.join(cache_dir, filename)
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data

def truncate_data(data, startDateStrTrunc, endDateStrTrunc):
  data_close_truncated = data['close'].loc[startDateStrTrunc:endDateStrTrunc].iloc[:-1] # reason for .iloc[:-1] -> the gather_data and gather_data_cached functions do not include endDateStrTrunc in the data, but using .loc slicing does include it. Therefore we remove one.
  data_open_truncated = data['open'].loc[startDateStrTrunc:endDateStrTrunc].iloc[:-1]
  data_high_truncated = data['high'].loc[startDateStrTrunc:endDateStrTrunc].iloc[:-1]
  data_low_truncated = data['low'].loc[startDateStrTrunc:endDateStrTrunc].iloc[:-1]
  data_vol_truncated = data['vol'].loc[startDateStrTrunc:endDateStrTrunc].iloc[:-1]
  data_yfinance_formatted_truncated = data['yfinance_formatted'].loc[startDateStrTrunc:endDateStrTrunc].iloc[:-1] # TODO: using .iloc is a bit wonky (only on this line, previous lines give exactly as expected) - it does give the correct shape for yfinance_formatted, but is not equal to the original function. 
  return {
      'close': data_close_truncated,
      'open': data_open_truncated,
      'high': data_high_truncated,
      'low': data_low_truncated,
      'vol': data_vol_truncated,
      'yfinance_formatted': data_yfinance_formatted_truncated
  }

def gather_data_cached_using_truncate(startDateStr, endDateStr, instrumentIds, cache_dir='.'):
    """
    Load the data from a cached file, given arguments matching gather_data.
    """
    # hardcoded to be the longest period that we have currently downloaded
    startDateStrEarliest = '2007-01-01'
    endDateStrLatest = '2024-12-31'
    filename = _get_filename(startDateStrEarliest, endDateStrLatest, instrumentIds)
    filepath = os.path.join(cache_dir, filename)
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    # now, truncate to be the shorter time period
    data_truncated = truncate_data(data, startDateStr, endDateStr)
    return data_truncated


def save_data(data, startDateStr, endDateStr, instrumentIds, cache_dir='.'):
    filename = _get_filename(startDateStr, endDateStr, instrumentIds)
    filepath = os.path.join(cache_dir, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def save_pairs_data_filtered(pairs_data_filtered, startDateStr, endDateStr, instrumentIds, cache_dir='.'):
    base_filename = _get_filename(startDateStr, endDateStr, instrumentIds)
    filename_pairs_data = base_filename.replace(".pkl", "_pairs_data_filtered.pkl")
    filepath = os.path.join(cache_dir, filename_pairs_data)
    with open(filepath, 'wb') as f:
        pickle.dump(pairs_data_filtered, f)

def gather_pairs_data_cached(startDateStr, endDateStr, instrumentIds, cache_dir='.'):
    base_filename = _get_filename(startDateStr, endDateStr, instrumentIds)
    filename_pairs_data = base_filename.replace(".pkl", "_pairs_data_filtered.pkl")
    filepath = os.path.join(cache_dir, filename_pairs_data)
    if os.path.exists(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    else:
        return None