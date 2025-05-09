import numpy as np
from statsmodels.tsa.stattools import coint
from statsmodels.tsa.stattools import MissingDataError
from tqdm import tqdm

def find_cointegrated_pairs(data, target_col='Close'):
    """
    Identifies cointegrated pairs of two assets from the input data. 
    In this case used for ETFs, but it should work for any time series.

    Parameters:
    -----------
    data : pandas.DataFrame
        A DataFrame where each column represents a time series.

    Returns:
    --------
    score_matrix : numpy.ndarray
        A 2D array where the element at (i, j) contains the cointegration score 
        for the pair of time series (i, j).
    pvalue_matrix : numpy.ndarray
        A 2D array where the element at (i, j) contains the p-value for the 
        cointegration test of the pair of time series (i, j).
    pairs : dict
        A dictionary where the keys are tuples of column names representing 
        cointegrated pairs, and the values are the results of the cointegration 
        test (score, p-value, and critical values).

    Notes:
    ------
    - The function is currently entirely sourced from the paper "Quantitative Trading Strategies using Deep Learning" by Stanford's Simerjot Kaur (2019). 
    - One change made is only using the 'Close' column. I saw that the original function looked at cointegration of all columns (Close, Open, High, Low, Volume) even though only the Close column is used for the analysis.
    - This function uses the Engle-Granger two-step cointegration test.


    Example:
    --------
    >>> import pandas as pd
    >>> data = pd.DataFrame({
    ...     'A': [1, 2, 3, 4, 5],
    ...     'B': [2, 4, 6, 8, 10],
    ...     'C': [5, 6, 7, 8, 9]
    ... })
    >>> score_matrix, pvalue_matrix, pairs = find_cointegrated_pairs(data)
    >>> print(pairs)
    {('A', 'B'): (score, pvalue, critical_values)}
    """
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    pairs = {}
    
    tickers = data.columns.get_level_values('Ticker').unique()
    # loop through all possible pairs of tickers
    keys = list(tickers)
    visited = 0
    total_pairs = len(keys) * (len(keys) - 1) // 2
    with tqdm(total=total_pairs, desc="Processing pairs") as pbar:
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                visited += 1
                S1 = data[keys[i]][target_col]
                S2 = data[keys[j]][target_col]
                try:
                    result = coint(S1, S2)
                except MissingDataError:
                    print(f"Missing data for pair: {keys[i]}, {keys[j]}")
                    pbar.update(1)
                    continue
                score = result[0]
                pvalue = result[1]
                score_matrix[i, j] = score
                pvalue_matrix[i, j] = pvalue
                if pvalue < 0.05:
                    pairs[(keys[i], keys[j])] = result
                pbar.update(1)
        print(f"Completed {visited} pairs")
        return score_matrix, pvalue_matrix, pairs


def find_cointegrated_pairs2(data):
    print('poep')
    return None, None, None

if __name__ == "__main__":
    import pandas as pd
    # Example data
    data = pd.DataFrame({
        ('A', 'Close'): [1, 2, 3, 4, 5],
        ('B', 'Close'): [2, 4, 6, 8, 10],
        ('C', 'Close'): [5, 6, 7, 8, 9],
        ('D', 'Close'): [10, 9, 8, 7, 6]
    })
    data.columns = pd.MultiIndex.from_tuples(data.columns, names=["Ticker", "Attribute"])

    # Run the function
    score_matrix, pvalue_matrix, pairs = find_cointegrated_pairs(data, target_col='Close')

    # Print results
    print("Score Matrix:")
    print(score_matrix)
    print("\nP-Value Matrix:")
    print(pvalue_matrix)
    print("\nCointegrated Pairs:")
    print(pairs)