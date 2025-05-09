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

def save_data(data, startDateStr, endDateStr, instrumentIds, cache_dir='.'):
    filename = _get_filename(startDateStr, endDateStr, instrumentIds)
    filepath = os.path.join(cache_dir, filename)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)
